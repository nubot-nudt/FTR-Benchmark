'''

'''
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from isaacgym_ext.interface.http_env import HttpAloneServerEnv
from isaacgym_ext.start import launch_isaacgym_env


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