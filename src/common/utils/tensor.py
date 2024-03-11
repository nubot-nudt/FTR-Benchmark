import numpy as np
import torch


def to_tensor(data):

    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)

    if isinstance(data, torch.Tensor):
        return data

    return torch.tensor(data)

    data = np.asarray(data)
    data = torch.from_numpy(data).float()
    return data

def to_numpy(data):

    if isinstance(data, np.ndarray):
        return data

    if isinstance(data, torch.Tensor):
        return data.numpy()

    return np.array(data)

def to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()

    if isinstance(data, torch.Tensor):
        return data.tolist()

    raise NotImplementedError()

def to_same_type(data, target):

    if isinstance(target, np.ndarray):
        return to_numpy(data)

    if isinstance(data, torch.Tensor):
        return to_tensor(data)

    raise NotImplementedError()

def to_same_dim(data, target):
    data_ = to_tensor(data)
    target = to_tensor(target)

    if target.dim() == 2 and data_.dim() == 1:
        return to_same_type(data_.view(-1, 1), data)

    if target.dim() == 2 and data_.dim() == 2:
        return to_same_type(data_, data)

    raise NotImplementedError()
