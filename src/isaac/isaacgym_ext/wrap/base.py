
import gym

class IsaacGymEnvWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)

    def __getattr__(self, name):
        return getattr(self.env, name)

    @property
    def task(self):
        return self.env._task

    @property
    def num_envs(self):
        return self.env.num_envs