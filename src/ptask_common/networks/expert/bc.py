# -*- coding: utf-8 -*-
"""
====================================
@File Name ：bc.py
@Time ： 2024/3/28 下午8:44
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import torch
from torch import nn
from torch import optim

from rl_games.algos_torch import network_builder


def train_soft_a2c_bc(expert_datas, epochs=100, device='cpu', lr=0.0005, batch_size=64, print_info=True, save_path=None,
                  save_freq=100):
    print(f'load {len(expert_datas)} (s, a) data')
    obses = torch.stack([e['obs'] for e in expert_datas]).to(device)
    actions = torch.stack([e['action'] for e in expert_datas]).to(device)

    actor_mlp_args = {
        'input_size': obses.shape[1],
        'units': [256, 256],
        'activation': 'elu',
        'norm_func_name': False,
        'dense_func': torch.nn.Linear,
        'd2rl': False,
        'norm_only_first_layer': False,
    }

    actor = network_builder.DiagGaussianActor(output_dim=8, log_std_bounds=[2, -4], **actor_mlp_args)
    actor.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(actor.parameters(), lr=lr)

    for epoch in range(epochs):
        for i in range(0, len(obses), batch_size):
            inputs = obses[i:i + batch_size]
            inputs += torch.normal(mean=0, std=0.005, size=inputs.size(), device=device)

            dist = actor(inputs)
            loss = criterion(dist.mean, actions[i:i + batch_size])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if print_info:
            print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

        if save_path is not None:
            if (epoch + 1) % save_freq == 0:
                path = save_path + f'/dg_actor_{epoch + 1}'
                torch.save(actor.state_dict(), path)
                print(f'save {path}')