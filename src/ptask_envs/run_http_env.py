

from ptask_envs.envs.interface.http_env import HttpAloneServerEnv
from ptask_envs.envs.start import launch_isaacgym_env


if __name__ == '__main__':
    def preprocess_func(config):
        config['headless'] = False
        config['task']['env']['numEnvs'] = 1

    with launch_isaacgym_env(preprocess_func) as ret:
        cfg_dict = ret['config']
        env = ret['env']

        env = HttpAloneServerEnv(env)

        print(env.observation_space)
        print(env.action_space)
        print('num_envs:', env.num_envs)

        env.run()