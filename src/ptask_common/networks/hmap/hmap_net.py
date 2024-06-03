# -*- coding: utf-8 -*-
"""
====================================
@File Name ：hmap_net.py
@Time ： 2024/3/20 下午4:47
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import torch
import torch.nn as nn
from ptask_common.networks.common import build_sequential_mlp


class HMapVAE(nn.Module):
    def __init__(self, latent_dim):
        super(HMapVAE, self).__init__()
        units = [128, 96, 64]
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(2, 2)),
            nn.ELU(),
            nn.Flatten(),
            build_sequential_mlp(120, units=units, output_size=2 * latent_dim, activation='elu'),
        )
        self.decoder = nn.Sequential(
            build_sequential_mlp(latent_dim, units=units[::-1], output_size=120, activation='elu'),
            nn.Unflatten(-1, (8, 5, 3)),
            nn.ConvTranspose2d(8, 16, kernel_size=(2, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 32, kernel_size=(4, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(32, 16, kernel_size=(4, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 1, kernel_size=(4, 2)),
            nn.ELU()
        )

    def encode(self, x):
        x = self.encoder(x)
        mu, logvar = torch.chunk(x, 2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


class HMapAE(nn.Module):
    def __init__(self, latent_dim):
        super(HMapAE, self).__init__()
        units = [latent_dim + 12, latent_dim + 6]
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=(4, 2)),
            nn.ELU(),
            nn.Conv2d(in_channels=8, out_channels=4, kernel_size=(2, 2)),
            nn.ELU(),
            nn.Flatten(),
            build_sequential_mlp(60, units=units, output_size=latent_dim, activation='elu'),
        )
        self.decoder = nn.Sequential(
            build_sequential_mlp(latent_dim, units=units[::-1], output_size=60, activation='elu'),
            nn.Unflatten(-1, (4, 5, 3)),
            nn.ConvTranspose2d(4, 8, kernel_size=(2, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(8, 16, kernel_size=(4, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(16, 8, kernel_size=(4, 2)),
            nn.ELU(),
            nn.ConvTranspose2d(8, 1, kernel_size=(4, 2)),
            nn.ELU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class HMapRec(nn.Module):
    def __init__(self, num_terrain_type):
        super(HMapRec, self).__init__()
        self.f = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(4, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 1), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_terrain_type)
        )

    def forward(self, x):
        x = torch.unsqueeze(x, 1)  # 增加一个通道维度
        x = self.f(x)
        return x
