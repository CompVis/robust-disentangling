import os, pickle
import itertools
import torch
import numpy as np
import edflow
from edflow import TemplateIterator, get_obj_from_str
from edflow.util import retrieve


def totorch(x, guess_image=True, device=None):
    if x.dtype == np.float64:
        x = x.astype(np.float32)
    x = torch.tensor(x)
    if guess_image and len(x.size()) == 4:
        x = x.transpose(2, 3).transpose(1, 2)
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    x = x.to(device)
    return x


def tonp(x, guess_image=True):
    try:
        if guess_image and len(x.shape) == 4:
            x = x.transpose(1, 2).transpose(2, 3)
        return x.detach().cpu().numpy()
    except AttributeError:
        return x


def get_learning_rate(config):
    if "learning_rate" in config:
        learning_rate = config["learning_rate"]
    elif "base_learning_rate" in config:
        learning_rate = config["base_learning_rate"]*config["batch_size"]
    else:
        learning_rate = 2.0e-4
    return learning_rate


class Iterator(TemplateIterator):
    """
    Pytorch base class to handle device and state. Adds optimizer and loss.
    Call update_lr() in train op for lr scheduling.

    Config parameters:
        - test_mode : boolean : Put model into .eval() mode.
        - no_restore_keys : string1,string2 : Submodels which should not be
                                              restored from checkpoint.
        - learning_rate : float : Learning rate of Adam
        - base_learning_rate : float : Learning_rate per example to adjust for
                                       batch size (ignored if learning_rate is present)
        - decay_start : float : Step after which learning rate is decayed to
                                zero.
        - loss : string : Import path of loss.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.test_mode = self.config.get("test_mode", False)
        if self.test_mode:
            self.model.eval()
        self.submodules = ["model"]
        self.no_restore_keys = retrieve(self.config, 'no_restore_keys', default='').split(',')

        self.learning_rate = get_learning_rate(self.config)
        self.logger.info("learning_rate: {}".format(self.learning_rate))
        if "loss" in self.config:
            self.loss = get_obj_from_str(self.config["loss"])(self.config)
            self.loss.to(self.device)
            self.submodules.append("loss")
            self.optimizer = torch.optim.Adam(
                    itertools.chain(self.model.parameters(), self.loss.parameters()),
                    lr=self.learning_rate,
                    betas=(0.5, 0.9))
            self.submodules.append("optimizer")

        self.num_steps = retrieve(self.config, "num_steps", default=0)
        self.decay_start = retrieve(self.config, "decay_start", default=self.num_steps)

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

    def get_state(self):
        state = dict()
        for k in self.submodules:
            state[k] = getattr(self, k).state_dict()
        return state

    def save(self, checkpoint_path):
        torch.save(self.get_state(), checkpoint_path)

    def restore(self, checkpoint_path):
        state = torch.load(checkpoint_path)
        keys = list(state.keys())
        for k in keys:
            if hasattr(self, k):
                if k not in self.no_restore_keys:
                    try:
                        missing, unexpected = getattr(self, k).load_state_dict(state[k], strict=False)
                        if missing:
                            self.logger.info("Missing keys for {}: {}".format(k, missing))
                        if unexpected:
                            self.logger.info("Unexpected keys for {}: {}".format(k, unexpected))
                    except TypeError:
                        self.logger.info(k)
                        try:
                            getattr(self, k).load_state_dict(state[k])
                        except ValueError:
                            self.logger.info("Could not load state dict for key {}".format(k))
                    else:
                        self.logger.info('Restored key `{}`'.format(k))
                else:
                    self.logger.info('Not restoring key `{}` (as specified)'.format(k))
            del state[k]

    def totorch(self, x, guess_image=True):
        return totorch(x, guess_image=guess_image, device=self.device)

    def tonp(self, x, guess_image=True):
        return tonp(x, guess_image=guess_image)


class Trainer(Iterator):
    def __init__(self, *args, **kwargs):
        config = args[0]
        self.eval_triplets = retrieve(config, "eval_triplets",
                                      default=False)
        if self.eval_triplets:
            self.triplet_functor = EvalTripletsFunctor(config)
            self.triplet_functor.set_iterator(self)
            self.callbacks = {"eval_op": self.triplet_functor.callbacks}

        super().__init__(*args, **kwargs)
        self.n_examples = 4
        self.fixed_examples = [
            self.datasets["validation"][i]["examples"]
            for i in np.random.RandomState(1).choice(len(self.datasets["validation"]),
                                                     self.n_examples)]
        self.fixed_x_pi = np.stack(e[0]["image"] for e in self.fixed_examples)
        self.fixed_x_pi = self.totorch(self.fixed_x_pi)
        self.fixed_x_al = np.stack(e[1]["image"] for e in self.fixed_examples)
        self.fixed_x_al = self.totorch(self.fixed_x_al)
        self.matrix_x_pi, self.matrix_x_al = self.make_matrix_pairs(
            self.fixed_x_pi, self.fixed_x_al)

    def make_matrix_pairs(self, x_pi, x_al):
        x_pi = x_pi[:self.n_examples,...]
        x_al = x_al[:self.n_examples,...]
        shape = tuple(x_pi.shape)
        x_pi = x_pi[None,...][self.n_examples*[0],...].reshape(
            self.n_examples*self.n_examples, *shape[1:])
        x_al = x_al[:,None,...][:,self.n_examples*[0],...].reshape(
            self.n_examples*self.n_examples, *shape[1:])
        return x_pi, x_al

    def step_op(self, model, **batch):
        x_target = self.totorch(batch["examples"][0]["image"])
        x_al = self.totorch(batch["examples"][1]["image"])
        x_pi = x_target

        p_pi, z_al = model.encode(x_pi, x_al)
        z_pi = p_pi.sample()
        x_out = model.decode(z_pi, z_al)

        auxpi_out = model.auxpi_decode(z_pi)
        auxal_out = model.auxal_decode(z_al)

        log_dict = {"images": dict(), "scalars": {}}

        loss, loss_train_op = self.loss(x_target, x_out, p_pi, z_al,
                                        auxpi_out, auxal_out,
                                        self.get_global_step(), log_dict)

        def train_op():
            loss_train_op()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.update_lr()

        def log_op():
            istrain = model.training
            model.eval()
            with torch.no_grad():
                batch_x_pi, batch_x_al = self.make_matrix_pairs(x_pi, x_al)
                batch_matrix = model(batch_x_pi, batch_x_al)
                fixed_matrix = model(self.matrix_x_pi, self.matrix_x_al)

                log_dict["images"]["batch_matrix"] = batch_matrix
                log_dict["images"]["fixed_matrix"] = fixed_matrix
                log_dict["images"]["x_pi"] = x_pi
                log_dict["images"]["x_al"] = x_al
                log_dict["images"]["x_out"] = x_out
                log_dict["images"]["auxpi_out"] = auxpi_out
                log_dict["images"]["auxal_out"] = auxal_out
            if istrain:
                model.train()
            for k in log_dict["images"]:
                log_dict["images"][k] = self.tonp(log_dict["images"][k])
            return log_dict

        ret_dict = {"train_op": train_op, "log_op": log_op}

        if self.eval_triplets:
            def eval_op():
                return self.triplet_functor(model, **batch)

            ret_dict["eval_op"] = eval_op

        return ret_dict


class EvalTripletsFunctor(object):
    def __init__(self, config):
        self.config = config

    def set_iterator(self, iterator):
        self.it = iterator

    @property
    def callbacks(self):
        return {"triplet_l2_error": triplet_mse_error}

    def __call__(self, model, **batch):
        x_target = self.it.totorch(batch["examples"][0]["image"])
        x_al = self.it.totorch(batch["examples"][1]["image"])
        x_pi = self.it.totorch(batch["examples"][2]["image"])

        x_out = model(x_pi, x_al)
        return {"labels": {"x_target": self.it.tonp(x_target),
                           "x_out": self.it.tonp(x_out)}}


def triplet_mse_error(root, data_in, data_out, config):
    per_example_mse = np.mean(np.square(data_out.labels["x_target"] -
                                        data_out.labels["x_out"]),
                              axis=(1,2,3))
    fpath = os.path.join(root, "per_example_mse.p")
    with open(fpath, "wb") as f:
        pickle.dump(per_example_mse, f)
    logger = edflow.get_logger("triplet_mse_error")
    logger.info(fpath)
    mse = np.mean(per_example_mse)
    std = np.std(per_example_mse)
    logger.info("mse: {:.4} +- {:.2}".format(mse, std))
    return {"scalars": {"mse": mse}}
