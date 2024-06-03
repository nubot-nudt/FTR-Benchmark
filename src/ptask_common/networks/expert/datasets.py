# -*- coding: utf-8 -*-
"""
====================================
@File Name ：datasets.py
@Time ： 2024/3/28 下午8:03
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import pickle
import pathlib
import torch
import pysnooper


def load_expert_demo(name):
    record_path = pathlib.Path('./data/datasets/expert') / (name + '.demo')
    with open(record_path, 'rb') as f:
        expert_demo = pickle.load(f)
    return expert_demo


# @pysnooper.snoop()
def preprocess_demo(demo, flipper_dt=2, action_mode='con_del_4'):

    demo_preprocessed = []

    for trj in demo:
        for d, next_d in zip(trj[:-1], trj[1:]):
            obs = [
                d['map'].flatten(),
                torch.clip(d['orient'][:2] / torch.pi, -1, 1),
                d['v'].unsqueeze(0),
                torch.clip(d['flipper'], -60 / 180 * torch.pi, 60 / 180 * torch.pi),
            ]

            if action_mode == 'con_del_4':
                action = torch.rad2deg(next_d['flipper'] - d['flipper']) / flipper_dt
            elif action_mode == 'pos_4':
                action = torch.clip(next_d['flipper'] / (torch.pi / 3), -1, 1)
                # action = -torch.ones((4, ))
            else:
                raise KeyError(f'{action_mode=} not impl')

            demo_preprocessed.append({
                'obs': torch.cat(obs).to(torch.float),
                'action': action.to(torch.float),
            })

    return demo_preprocessed
