# -*- coding: utf-8 -*-
"""
====================================
@File Name ：omega_conf.py
@Time ： 2024/10/8 下午1:57
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from functools import reduce
from operator import add, mul
import os

from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eq", lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver("contains", lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver("if", lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver("resolve_default", lambda default, arg: default if arg == "" else arg)
OmegaConf.register_new_resolver("len", lambda x: len(x))
OmegaConf.register_new_resolver("cat", lambda x, y: str(x) + str(y))
OmegaConf.register_new_resolver("add", lambda *args: reduce(add, args, 0))
OmegaConf.register_new_resolver("mul", lambda *args: reduce(mul, args, 1))
OmegaConf.register_new_resolver("min", min)
OmegaConf.register_new_resolver("max", max)
OmegaConf.register_new_resolver("basename", os.path.basename)