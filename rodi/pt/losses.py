from edflow.util import retrieve
import torch.nn as nn
import torch

from rodi.models import MIEstimator
from rodi.iterator import get_learning_rate


class MI(nn.Module):
    def __init__(self, config, prefix=""):
        super().__init__()
        self.config = config
        self.prefix = prefix

        self.lr_factor = retrieve(config, "MI/lr_factor", default=2.0)
        self.learning_rate = get_learning_rate(config)
        self.num_steps = retrieve(self.config, "num_steps", default=0)
        self.decay_start = retrieve(self.config, "decay_start", default=self.num_steps)

        self.estimator = MIEstimator(config)
        self.optimizer = torch.optim.Adam(
            self.estimator.parameters(),
            lr=self.lr_factor*self.learning_rate,
            betas=(0.5, 0.9))

        self.register_buffer("avg_loss", torch.zeros(size=()))
        self.register_buffer('count_loss', torch.tensor(0, dtype=torch.int))
        self.register_buffer("avg_acc", torch.zeros(size=()))
        self.register_buffer('count_acc', torch.tensor(0, dtype=torch.int))
        self.register_buffer("avg_mi", torch.zeros(size=()))
        self.register_buffer('count_mi', torch.tensor(0, dtype=torch.int))

    def update_ema(self, name, value):
        value = value.detach() # detach from graph or it will build up
        ema = getattr(self, "avg_"+name)
        count = getattr(self, "count_"+name)
        count.add_(1)
        decay = 0.99
        decayp = decay**count.float()
        new_ema = ((decay-decayp)*ema + (1.0-decay)*value)/(1.0-decayp)
        ema.copy_(new_ema)

    def get_decay_factor(self):
        alpha = 1.0
        if self.num_steps > self.decay_start:
            alpha = 1.0 - np.clip(
                (self.get_global_step() - self.decay_start) /
                (self.num_steps - self.decay_start),
                0.0, 1.0)
        return alpha

    def update_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.get_decay_factor()*self.learning_rate

    def parameters(self):
        """Exclude estimator from parameters."""
        ps = super().parameters()
        exclude = set(self.estimator.parameters())
        ps = (p for p in ps if not p in exclude)
        return ps

    def forward(self, z_pi, z_al, global_step, log_dict):
        l_joint = self.estimator(z_pi, z_al)
        l_marginal = self.estimator(z_pi, torch.flip(z_al, dims=(0,)))

        loss = 0.5*torch.mean(
            torch.nn.functional.softplus(-l_joint) +
            torch.nn.functional.softplus(l_marginal))

        acc = 0.5*(
            torch.mean((l_joint>0).float()) +
            torch.mean((l_marginal<0).float()))

        mi = torch.mean(l_joint)

        def train_op():
            # update estimator
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # update ema
            self.update_ema("loss", loss)
            self.update_ema("acc", acc)
            self.update_ema("mi", mi)
            # update lr
            self.update_lr()

        log_dict["scalars"][self.prefix+"loss"] = loss
        log_dict["scalars"][self.prefix+"acc"] = acc
        log_dict["scalars"][self.prefix+"mi"] = mi
        log_dict["scalars"][self.prefix+"avg_loss"] = self.avg_loss
        log_dict["scalars"][self.prefix+"avg_acc"] = self.avg_acc
        log_dict["scalars"][self.prefix+"avg_mi"] = self.avg_mi

        return mi, train_op


class Elbo(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kl_weight = retrieve(config, "Elbo/kl_weight", default=1.0)
        self.mi_estimator = MI(config, prefix="mi_")

    def forward(self, x_target, x_out, p_pi, z_al, auxpi_out, auxal_out,
                global_step, log_dict, kl_weight=None):
        kl_weight = kl_weight if kl_weight is not None else self.kl_weight

        rec_loss = (x_target - x_out)**2
        rec_loss = torch.sum(rec_loss) / rec_loss.shape[0]

        aux_loss = 0.5*((x_target - auxpi_out)**2 + (x_target - auxal_out)**2)
        aux_loss = torch.sum(aux_loss) / aux_loss.shape[0]

        kl_loss = p_pi.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

        mi, mi_train_op = self.mi_estimator(p_pi.sample(), z_al,
                                            global_step, log_dict)

        loss = rec_loss + kl_weight*kl_loss + aux_loss

        log_dict["scalars"]["loss"] = loss
        log_dict["scalars"]["rec_loss"] = rec_loss
        log_dict["scalars"]["aux_loss"] = aux_loss
        log_dict["scalars"]["kl_loss"] = kl_loss

        return loss, mi_train_op


class MIMin(nn.Module):
    """
    Minimize ELBO subject to MI <= eps
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.elbo = Elbo(config)
        self.mi1_estimator = MI(config, prefix="mi1_")
        self.lambda_init = retrieve(config, "MIMin/lambda_init", default=0.0)
        self.register_buffer("lambda_", torch.ones(size=())*self.lambda_init)
        self.mu = retrieve(config, "MIMin/mu", default=0.1)
        self.eps = retrieve(config, "MIMin/eps", default=0.1)
        self.warmup_it = retrieve(config, "MIMin/warmup_it", default=1000)

    def forward(self, x_target, x_out, p_pi, z_al, auxpi_out, auxal_out, global_step, log_dict):
        loss, mi_train_op = self.elbo(x_target, x_out, p_pi, z_al, auxpi_out,
                                      auxal_out, global_step, log_dict)

        z_pi = p_pi.sample()
        mi1, mi1_train_op = self.mi1_estimator(z_pi, z_al.detach(),
                                               global_step, log_dict)
        constraint = mi1 - self.eps
        # make a copy of self.lambda_ because it will be modified in train op
        lambda_ = self.lambda_.clone()
        active = (self.mu*constraint + lambda_ >= 0).detach().float()
        # use warmup period to avoid abstruse values for the constraint
        warm = global_step >= self.warmup_it
        loss_mi = warm*active*(lambda_*constraint + 0.5*self.mu*constraint**2)

        loss = loss + loss_mi

        log_dict["scalars"]["loss"] = loss
        log_dict["scalars"]["loss_mi"] = loss_mi
        log_dict["scalars"]["constraint"] = constraint
        log_dict["scalars"]["lambda_"] = self.lambda_

        def train_op():
            mi_train_op()
            mi1_train_op()
            # update lambda_
            if warm:
                new_lambda = self.lambda_ + self.mu*constraint
                new_lambda = torch.clamp(new_lambda, min=0.0)
                self.lambda_.copy_(new_lambda.detach())

        return loss, train_op


class RoDi(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.elbo = Elbo(config)
        self.mi1_estimator = MI(config, prefix="mi1_")
        self.lambda_init = retrieve(config, "MIMin/lambda_init", default=0.0)
        self.register_buffer("lambda_", torch.ones(size=())*self.lambda_init)
        self.mu = retrieve(config, "MIMin/mu", default=0.1)
        self.eps = retrieve(config, "MIMin/eps", default=0.1)
        self.warmup_it = retrieve(config, "MIMin/warmup_it", default=1000)

        self.mi2_estimator = MI(config, prefix="mi2_")
        self.gamma_init = retrieve(config, "RoDi/gamma_init", default=self.elbo.kl_weight)
        self.register_buffer("gamma", torch.ones(size=())*self.gamma_init)
        self.b_gamma = retrieve(config, "RoDi/b_gamma", default=0.01)
        self.l_gamma = retrieve(config, "RoDi/l_gamma", default=0.01)

    def forward(self, x_target, x_out, p_pi, z_al, auxpi_out, auxal_out, global_step, log_dict):
        loss, mi_train_op = self.elbo(x_target, x_out, p_pi, z_al, auxpi_out,
                                      auxal_out, global_step, log_dict,
                                      kl_weight=self.gamma.clone().detach())

        z_pi = p_pi.sample()
        mi1, mi1_train_op = self.mi1_estimator(z_pi, z_al.detach(),
                                               global_step, log_dict)
        constraint = mi1 - self.eps
        # make a copy of self.lambda_ because it will be modified in train op
        lambda_ = self.lambda_.clone()
        active = (self.mu*constraint + lambda_ >= 0).detach().float()
        # use warmup period to avoid abstruse values for the constraint
        warm = global_step >= self.warmup_it
        loss_mi = warm*active*(lambda_*constraint + 0.5*self.mu*constraint**2)

        loss = loss + loss_mi

        # train mi2_estimator
        mi2, mi2_train_op = self.mi2_estimator(z_pi.detach(), z_al.detach(),
                                               global_step, log_dict)
        mi_diff = log_dict["scalars"]["mi2_avg_mi"] - log_dict["scalars"]["mi1_avg_mi"] - self.b_gamma

        log_dict["scalars"]["loss"] = loss
        log_dict["scalars"]["loss_mi"] = loss_mi
        log_dict["scalars"]["constraint"] = constraint
        log_dict["scalars"]["lambda_"] = self.lambda_
        log_dict["scalars"]["gamma"] = self.gamma
        log_dict["scalars"]["mi_diff"] = mi_diff

        def train_op():
            mi_train_op()
            mi1_train_op()
            mi2_train_op()
            # update lambda_ and gamma
            if warm:
                new_lambda = self.lambda_ + self.mu*constraint
                new_lambda = torch.clamp(new_lambda, min=0.0)
                self.lambda_.copy_(new_lambda.detach())

                new_gamma = self.gamma + self.l_gamma*mi_diff
                new_gamma = torch.clamp(new_gamma, min=0.0)
                self.gamma.copy_(new_gamma.detach())

        return loss, train_op
