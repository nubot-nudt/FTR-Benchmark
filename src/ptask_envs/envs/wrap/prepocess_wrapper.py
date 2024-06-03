import gym
from gym import Env


class ObsUnpackDictWrapper(gym.ObservationWrapper):

    def __init__(self, env: Env, cfg, key='obs'):
        super().__init__(env)
        self._unpack_dict_key = key

    def observation(self, obs):
        return obs[self._unpack_dict_key]


