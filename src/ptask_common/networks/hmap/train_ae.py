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
from torch import nn
from torch import optim
from collections import deque

from .hmap_net import HMapAE


def train_ae(data, latent_dim, epochs=1000, batch_size=1024, lr=1e-5, device='cuda:0', save_path=None,
             save_freq=100, print_info=False):
    model = HMapAE(latent_dim)

    data = data.to(device)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, eps=1e-10, weight_decay=1e-7)
    # optimizer = optim.RMSprop(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            inputs = data[i:i + batch_size]
            inputs = inputs.view(-1, 1, 15, 7)
            inputs_noise = inputs + torch.normal(mean=0, std=0.005, size=inputs.size(), device=device)

            outputs = model(inputs_noise)
            loss = criterion(outputs, inputs_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_info:
            m = 0
            for p in model.parameters():
                m = max(torch.max(p), m)
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}, Max Param: {m.item()}')

        if save_path is not None:
            if (epoch + 1) % save_freq == 0:
                torch.save(model.state_dict(), save_path + f'/hmap_ae_{epoch + 1}')

    return loss.item()
