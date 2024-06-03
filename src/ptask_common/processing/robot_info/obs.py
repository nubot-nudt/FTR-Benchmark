import numpy as np
import torch

from ptask_common.utils.tensor import to_tensor, to_same_type


class ObsSlice:
    def __init__(self, obs_space: dict):
        self.names = obs_space.keys()
        self.lengths = [obs_space[i] if isinstance(obs_space[i], int) else obs_space[i]['size'] for i in self.names]
        self.indexes = np.cumsum(self.lengths)
        self.index_dict = {n: i for n, i in zip(self.names, self.indexes)}
        self.length_dict = {n: i for n, i in zip(self.names, self.lengths)}
    def obs_dict_to_array(self, obs: dict):
        return to_same_type(torch.cat([to_tensor(obs[n]) for n in self.names], dim=-1), obs[self.names[0]])

    def obs_array_to_dict(self, obs) -> dict:
        obs = to_tensor(obs)

        if obs.dim() == 1:
            return {n: to_same_type(obs[i-l:i], obs) for n, l, i in zip(self.names, self.lengths, self.indexes)}
        else:
            return {n: to_same_type(obs[:, i - l:i], obs) for n, l, i in zip(self.names, self.lengths, self.indexes)}

    def get_array_by_name(self, obs, name):
        obs_t = to_tensor(obs)

        i = self.index_dict[name]
        if obs.dim() == 1:
            return to_same_type(obs_t[i - self.length_dict[name]:i], obs)
        else:
            return to_same_type(obs_t[:, i - self.length_dict[name]:i], obs)

    def set_array_by_name(self, obs, name, array):
        obs_t = to_tensor(obs)

        i = self.index_dict[name]
        if obs.dim() == 1:
            obs_t[i - self.length_dict[name]:i] = array
        else:
            obs_t[:, i - self.length_dict[name]:i] = array


def obs_dict_to_array(obs_space, obs: dict):
    return ObsSlice(obs_space).obs_dict_to_array(obs)

def obs_array_to_dict(obs_space, obs) -> dict:
    return ObsSlice(obs_space).obs_array_to_dict(obs)