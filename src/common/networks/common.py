import torch
import torch.nn as nn


class ActivationFactory:
    _registered_activations = {}

    @classmethod
    def register_builder(cls, name, func):
        cls._registered_activations[name] = func

    @classmethod
    def get_activation_by_name(cls, name):
        return cls._registered_activations[name]()


ActivationFactory.register_builder('relu', lambda **kwargs: nn.ReLU(**kwargs))
ActivationFactory.register_builder('tanh', lambda **kwargs: nn.Tanh(**kwargs))
ActivationFactory.register_builder('sigmoid', lambda **kwargs: nn.Sigmoid(**kwargs))
ActivationFactory.register_builder('elu', lambda **kwargs: nn.ELU(**kwargs))
ActivationFactory.register_builder('selu', lambda **kwargs: nn.SELU(**kwargs))
ActivationFactory.register_builder('swish', lambda **kwargs: nn.SiLU(**kwargs))
ActivationFactory.register_builder('gelu', lambda **kwargs: nn.GELU(**kwargs))
ActivationFactory.register_builder('softplus', lambda **kwargs: nn.Softplus(**kwargs))
ActivationFactory.register_builder('None', lambda **kwargs: nn.Identity())


def build_sequential_mlp(input_size, units, output_size=None, activation='relu'):
    if len(units) == 0:
        return nn.Linear(input_size, output_size)

    # 添加输入层到第一个隐藏层
    layers = [nn.Linear(input_size, units[0]), ActivationFactory.get_activation_by_name(activation)]

    # 添加隐藏层
    for i in range(1, len(units)):
        layers.append(nn.Linear(units[i - 1], units[i]))
        layers.append(ActivationFactory.get_activation_by_name(activation))

    if output_size is not None:
        if output_size > 0:
            layers.append(nn.Linear(units[-1], output_size))
        else:
            layers.append(nn.Linear(units[-1], units[-1]))

    # 构建模型
    model = nn.Sequential(*layers)

    return model


