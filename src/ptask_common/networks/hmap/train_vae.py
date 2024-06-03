# -*- coding: utf-8 -*-
"""
====================================
@File Name ：train_ae.py
@Time ： 2024/3/27 下午8:51
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from .hmap_net import HMapVAE


def train_vae(data, latent_dim, epochs=1000, batch_size=1024, lr=1e-5, device='cuda:0', save_path=None,
              save_freq=100, print_info=False):
    model = HMapVAE(latent_dim)

    data = data.to(device)
    model.to(device)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-6)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            inputs = data[i:i + batch_size]
            inputs = inputs.view(-1, 1, 15, 7)
            inputs_noise = inputs + torch.normal(mean=0, std=0.005, size=inputs.size(), device=device)

            recon_x, mu, logvar = model(inputs_noise)

            BCE = F.mse_loss(recon_x, inputs_noise)
            KLD = -torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            loss = BCE + KLD * 1e-5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_info:
            m = 0
            for p in model.parameters():
                m = max(torch.max(p), m)
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, BCE: {BCE}, KLD: {KLD}, Max Param: {m.item()}')

        if save_path is not None:
            if (epoch + 1) % save_freq == 0:
                torch.save(model.state_dict(), save_path + f'/hmap_vae_{epoch + 1}')

    return loss.item()



