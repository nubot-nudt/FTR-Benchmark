# -*- coding: utf-8 -*-
"""
====================================
@File Name ：quat.py
@Time ： 2024/3/27 上午10:23
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import torch
from omni.isaac.core.utils.torch.rotations import *

@torch.jit.script
def quat_apply_yaw(quat, vec):
    quat_yaw = quat.clone().view(-1, 4)
    quat_yaw[:, 1:3] = 0.0
    quat_yaw = normalize(quat_yaw)
    return quat_apply(quat_yaw, vec)