# -*- coding: utf-8 -*-
"""
====================================
@File Name ：train_recognise.py
@Time ： 2024/4/2 下午3:13
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import torch
from torch import nn
from torch import optim

from .hmap_net import HMapRec

def train_recongnize(datas, labels, terrain_types, epochs=1000, batch_size=1024, lr=0.005, device='cuda', save_path=None,
                     save_freq=100, print_info=False, test=False, L1_lambda=0.0001):
    # 实例化模型
    model = HMapRec(len(terrain_types))

    datas = datas.to(device)
    labels = labels.to(device)
    model.to(device)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 将数据划分为训练集和测试集
    train_ratio = 0.8
    train_size = int(len(datas) * train_ratio)

    train_data, test_data = datas[:train_size], datas[train_size:]
    train_labels, test_labels = labels[:train_size], labels[train_size:]

    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(0, len(train_data), batch_size):
            inputs, labels = train_data[i:i + batch_size], train_labels[i:i + batch_size]
            inputs += torch.normal(mean=0, std=0.005, size=inputs.size(), device=device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            for p in model.parameters():
                loss += L1_lambda * torch.norm(p, 1)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if print_info:
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_data)}')

        if save_path is not None:
            if (epoch + 1) % save_freq == 0:
                torch.save(model.state_dict(), save_path + f'/hmap_rec_{epoch + 1}')

    torch.save(model.state_dict(), save_path + f'/hmap_rec')

    if test:
        # 在测试集上评估模型
        correct = 0
        total = 0

        with torch.no_grad():
            for i in range(0, len(test_data), batch_size):
                inputs, labels = test_data[i:i + batch_size], test_labels[i:i + batch_size]
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy: {100 * correct / total}%')
