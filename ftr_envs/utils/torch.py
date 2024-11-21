# -*- coding: utf-8 -*-
"""
====================================
@File Name ：torch.py
@Time ： 2024/10/12 下午8:50
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import torch


def add_noise(data, std):
    size = data.size()
    noise = torch.normal(0, std=std, size=size)
    return data + noise


def rand_range(range_, *args, **kargs):
    """

    :param range_: (_min, _max) value
    :param args: same torch.rand
    :param kargs: same tor.rand
    :return:
    """
    return torch.rand(*args, **kargs) * (
            range_[1] - range_[0]
    ) + range_[0]
