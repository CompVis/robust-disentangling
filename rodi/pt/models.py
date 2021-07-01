import torch
from torch import nn


class Distribution(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 10.0)
        self.std = torch.exp(0.5*self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self, temperature=1.0):
        x = self.mean + temperature*self.std*torch.randn(self.mean.shape).to(self.mean)
        return x

    def kl(self, other=None):
        if other is None:
            return 0.5*torch.sum(torch.pow(self.mean, 2)
                    + self.var - 1.0 - self.logvar,
                    dim=[1,2,3])
        else:
            return 0.5*torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1,2,3])


def DownBlock(in_channels, out_channels):
    return nn.Conv2d(in_channels=in_channels,
                     out_channels=out_channels,
                     kernel_size=4,
                     padding=1,
                     stride=2)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=4*out_channels,
                              kernel_size=3,
                              padding=1)
        self.up = nn.PixelShuffle(2)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.act = nn.ReLU()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        r = self.act(x)
        r = self.conv(x)
        return x+r


class Encoder(nn.Module):
    def __init__(self, in_channels, expansion, max_channels, n_down,
                 n_res, out_channels):
        super().__init__()
        in_channels = in_channels
        ex_channels = min(max_channels, expansion)
        self.sub_layers = nn.ModuleList()
        self.sub_layers.append(nn.Conv2d(in_channels=in_channels,
                                         out_channels=ex_channels,
                                         kernel_size=3,
                                         padding=1))

        for i in range(n_down):
            in_channels = ex_channels
            ex_channels = min(max_channels, ex_channels*2**(i+1))
            self.sub_layers.append(ResBlock(in_channels=in_channels))
            self.sub_layers.append(DownBlock(in_channels=in_channels,
                                             out_channels=ex_channels))

        for _ in range(n_res):
            self.sub_layers.append(ResBlock(in_channels=ex_channels))

        self.sub_layers.append(nn.ReLU())
        self.sub_layers.append(nn.Conv2d(in_channels=ex_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         padding=1))

    def forward(self, x):
        for layer in self.sub_layers:
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, expansion, max_channels, n_up,
                 n_res, out_channels):
        super().__init__()
        in_channels = in_channels
        ex_channels = min(max_channels, expansion*2**n_up)
        self.sub_layers = nn.ModuleList()
        self.sub_layers.append(nn.Conv2d(in_channels=in_channels,
                                         out_channels=ex_channels,
                                         kernel_size=3,
                                         padding=1))

        for _ in range(n_res):
            self.sub_layers.append(ResBlock(in_channels=ex_channels))

        for i in range(n_up):
            in_channels = ex_channels
            ex_channels = min(max_channels, expansion*2**(n_up-1-i))
            self.sub_layers.append(ResBlock(in_channels=in_channels))
            self.sub_layers.append(UpBlock(in_channels=in_channels,
                                           out_channels=ex_channels))


        self.sub_layers.append(ResBlock(in_channels=ex_channels))
        self.sub_layers.append(nn.Conv2d(in_channels=ex_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         padding=1))

    def forward(self, x):
        for layer in self.sub_layers:
            x = layer(x)
        return x


class VAE(nn.Module):
    """This differs slightly from the architecture in the paper: Decoder
    applies residual blocks to concatenation of both codes instead of handling
    them seperately."""
    def __init__(self, config):
        super().__init__()
        self.pi_encoder = Encoder(in_channels=3,
                                  expansion=16,
                                  max_channels=256,
                                  n_down=4,
                                  n_res=5,
                                  out_channels=2*16)

        self.al_encoder = Encoder(in_channels=3,
                                  expansion=16,
                                  max_channels=256,
                                  n_down=4,
                                  n_res=5,
                                  out_channels=16)

        self.decoder = Decoder(in_channels=2*16,
                               expansion=16,
                               max_channels=256,
                               n_up=4,
                               n_res=4,
                               out_channels=3)
        self.auxal_decoder = Decoder(in_channels=16,
                                     expansion=16,
                                     max_channels=256,
                                     n_up=4,
                                     n_res=4,
                                     out_channels=3)
        self.auxpi_decoder = Decoder(in_channels=16,
                                     expansion=16,
                                     max_channels=256,
                                     n_up=4,
                                     n_res=4,
                                     out_channels=3)

    def encode(self, x_pi, x_al):
        p_pi = Distribution(self.pi_encoder(x_pi))
        z_al = self.al_encoder(x_al)
        return p_pi, z_al

    def decode(self, z_pi, z_al):
        z = torch.cat([z_pi, z_al], dim=1)
        x_out = self.decoder(z)
        return x_out

    def auxal_decode(self, z_al):
        return self.auxal_decoder(z_al)

    def auxpi_decode(self, z_pi):
        z_pi = z_pi.detach()
        return self.auxpi_decoder(z_pi)

    def forward(self, x_pi, x_al):
        p_pi, z_al = self.encode(x_pi, x_al)
        temp = 1.0 if self.training else 0.0
        z_pi = p_pi.sample()
        x_out = self.decode(z_pi, z_al)
        return x_out


class ResMLP(nn.Module):
    def __init__(self, in_channels, expansion, n_res, out_channels):
        super().__init__()
        in_channels = in_channels
        ex_channels = expansion
        self.sub_layers = nn.ModuleList()
        self.sub_layers.append(nn.Conv2d(in_channels=in_channels,
                                         out_channels=ex_channels,
                                         kernel_size=3,
                                         padding=1))

        for _ in range(n_res):
            self.sub_layers.append(ResBlock(in_channels=ex_channels))


        self.sub_layers.append(nn.ReLU())
        self.sub_layers.append(nn.Conv2d(in_channels=ex_channels,
                                         out_channels=out_channels,
                                         kernel_size=3,
                                         padding=1))

    def forward(self, x):
        for layer in self.sub_layers:
            x = layer(x)
        return x


class MIEstimator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net1 = ResMLP(in_channels=16,
                           expansion=512,
                           n_res=4,
                           out_channels=16)
        self.net2 = ResMLP(in_channels=16,
                           expansion=512,
                           n_res=4,
                           out_channels=16)

    def forward(self, z_pi, z_al):
        z1 = self.net1(z_pi)
        z2 = self.net2(z_al)
        l = torch.sum(z1*z2, dim=(1,2,3))
        return l
