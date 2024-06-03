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


def build_sequential_mlp(input_size, units, output_size=None, activation='relu', norm_func_name='none'):
    if len(units) == 0:
        return nn.Linear(input_size, output_size)

    units_ = [input_size] + units
    if output_size is not None:
        units_ += [output_size]

    layers = []
    for isize, osize in zip(units_[:-1], units_[1:]):
        layers.append(nn.Linear(isize, osize))
        layers.append(ActivationFactory.get_activation_by_name(activation))

        if norm_func_name == 'layer_norm':
            layers.append(torch.nn.LayerNorm(osize))
        elif norm_func_name == 'batch_norm':
            layers.append(torch.nn.BatchNorm1d(osize))

    model = nn.Sequential(*layers)
    return model


