# -*- coding: utf-8 -*-
"""
====================================
@File Name ：patterns.py
@Time ： 2024/5/22 下午12:32
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


class BaseFactory:
    _registered = {}

    def __init__(self, parent_cls=None, func_suffix=None):
        if parent_cls is None and func_suffix is None:
            raise RuntimeError('parent_cls and func_suffix cannot be None at the same time')

        if parent_cls is not None:
            self._type = 'cls'
            self.parent_cls = parent_cls

        if func_suffix is not None:
            self._type = 'func'
            self.func_suffix = func_suffix

    def register(self, name, obj):
        if name not in self._registered:
            self._registered[name] = obj
        else:
            raise KeyError(f'{name} already exists')

    def build(self, name, *args, **kwargs):
        return self._registered[name](*args, **kwargs)

    def get_names(self):
        return list(self._registered.keys())
