# -*- coding: utf-8 -*-
"""
====================================
@File Name ：torch.py
@Time ： 2024/5/22 上午11:52
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

from functools import partial
import torch


def init_load_device(device):
    torch.load = partial(torch.load, map_location=torch.device(device))