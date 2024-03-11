
import torch
from torch import nn

from .common import build_sequential_mlp

class SNN(nn.Module):

    def __init__(self, input_size, latent_dim, units, output_size, activation='relu', bilinear_integration=False):
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.units = units
        self.output_size = output_size
        self.bilinear_integration = bilinear_integration

        self.mlp = build_sequential_mlp(self.mlp_input_size, units, output_size, activation)

    def forward(self, x, latent_var):
        if self.bilinear_integration:
            extended_x = torch.cat(
                [x, latent_var, (x.unsqueeze(dim=-1) * latent_var.unsqueeze(dim=-2)).view(-1, self.input_size * self.latent_dim)], dim=1)
        else:
            extended_x = torch.concat([x, latent_var], dim=1)
        return self.mlp(extended_x)

    @property
    def mlp_input_size(self):
        return self.input_size + self.latent_dim + 0 if self.bilinear_integration else self.input_size * self.latent_dim


