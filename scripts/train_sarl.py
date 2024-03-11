'''

'''
import sys, os

import gym

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from isaacgym_ext.start import launch_isaacgym_env

from sarl.algorithms.rl.ppo.ppo import PPO
from sarl.algorithms.rl.sac.sac import SAC
from sarl.algorithms.rl.trpo.trpo import TRPO
from sarl.algorithms.rl.td3.td3 import TD3
from sarl.algorithms.rl.ddpg.ddpg import DDPG


def process_sarl(env, cfg_dict):
    cfg_train = cfg_dict['train']['params']

    learn_cfg = cfg_train["learn"]
    is_testing = learn_cfg["test"]

    # is_testing = True
    # Override resume and testing flags if they are passed as parameters.
    if cfg_dict['checkpoint'] != "":
        is_testing = True
        chkpt_path = cfg_dict['checkpoint']

    if cfg_dict['max_iterations'] != -1:
        cfg_train["learn"]["max_iterations"] = cfg_dict['checkpoint']

    logdir = cfg_train["learn"]['full_experiment_name']

    class Wrap(gym.Wrapper):

        def reset(self):
            return self.env.reset()['obs']

        def step(self, action):
            obs, rew, done, info = self.env.step(action)
            return obs['obs'], rew, done, info
        def get_state(self):
            return self.env._task.obs_buf

    """Set up the algo system for training or inferencing."""
    model = eval(cfg_train['algo']['name'].upper())(vec_env=Wrap(env),
              cfg_train = cfg_train,
              device=cfg_train['learn']['device'],
              sampler=learn_cfg.get("sampler", 'sequential'),
              log_dir=logdir,
              is_testing=is_testing,
              print_log=learn_cfg["print_log"],
              apply_reset=False,
              asymmetric=(env.num_states > 0)
              )

    # ppo.test("/home/hp-3070/logs/demo/scissors/ppo_seed0/model_6000.pt")
    if is_testing and cfg_dict['checkpoint'] != "":
        print("Loading model from {}".format(chkpt_path))
        model.test(chkpt_path)
    elif cfg_dict['checkpoint'] != "":
        print("Loading model from {}".format(chkpt_path))
        model.load(chkpt_path)

    return model


if __name__ == '__main__':
    with launch_isaacgym_env() as ret:
        cfg_dict = ret['config']
        env = ret['env']

        if cfg_dict['train']['params']['algo']['name'] in ["ppo", "ddpg", "sac", "td3", "trpo"]:

            sarl = process_sarl(env, cfg_dict)

            iterations = cfg_dict["max_iterations"]

            sarl.run(num_learning_iterations=int(iterations),
                     log_interval=cfg_dict['train']['params']["learn"]["save_interval"])

        else:
            raise NotImplementedError()


