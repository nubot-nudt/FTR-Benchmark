# -*- coding: utf-8 -*-
"""
====================================
@File Name ：datasets.py
@Time ： 2024/3/27 下午9:00
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import pickle
import os
import torch


def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


def load_dataset():
    # 加载数据集
    base_path = 'data/datasets/height_map'
    terrain_types = os.listdir(base_path)
    datas = [load_data(f'{base_path}/{t}') for t in terrain_types]
    print(f'{terrain_types=}')

    # 数据预处理
    for data in datas:
        for i in range(len(data)):
            data[i] -= torch.min(data[i])

    # 将数据处理成 PyTorch 可用的张量
    data_tensors = [torch.stack(data).float() for data in datas]

    # 创建标签
    labels = [torch.ones(len(data), dtype=torch.long) * i for i, data in enumerate(datas)]

    # 合并数据和标签
    data = torch.cat(data_tensors, dim=0)
    labels = torch.cat(labels, dim=0)

    # 随机打乱数据
    shuffle_indices = torch.randperm(len(data))
    data = data[shuffle_indices]
    labels = labels[shuffle_indices]
    return data, labels, terrain_types
