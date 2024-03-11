import os

from autonomic.executor.rlgames import RlgamesPredictor
from autonomic.executor.http import HttpPredictor

from isaacgym_ext.start import launch_isaacgym_env
from isaacgym_ext.interface.numpy_env import NumpyAloneEnv
from isaacgym_ext.interface.http_pub import HttpPubEnv
from utils.tensor import to_numpy


from .base import Experiment

class ExerimentRunner():
    def __init__(self, cfg_dict, env, experiment: Experiment, executor=None, epoch=10):
        self.experiment = experiment
        self.experiment.before_init(cfg_dict, env)

        self.env = env
        self.cfg = cfg_dict

        self.epoch = epoch

        if executor is None:
            self.epoch = cfg_dict['experiment_config'].get('epoch', -1)
            self.executor = RlgamesPredictor.load_model(
                **cfg_dict['experiment_config']['executor'],
                action_space=env.action_space, observation_space=env.observation_space

            )
        else:
            self.executor = executor

        os.makedirs('./data/experiment', exist_ok=True)

        self.experiment.after_init(self)

    def start(self):
        self.experiment.on_start()

        for i in range(1, self.epoch + 1):
            self.experiment.before_episode(i)
            self.run_once(i)
            self.experiment.after_episode(i)

        self.experiment.on_end()

    def run_once(self, epoch):

        obs = self.env.reset()
        step = 1
        all_reward = 0

        while True:
            action = self.executor.get_action(obs)
            self.experiment.before_step(step, obs, action)
            obs, reward, done, info = self.env.step(to_numpy(action))
            self.experiment.after_step(step, obs, reward, done, info)
            all_reward += reward

            step += 1
            if done:
                print(f'epoch: {epoch}, reward: {all_reward}')
                break


def create_runner(experiment: Experiment, is_http_pub=False, executor=None):
    def preprocess_func(config):
        config['headless'] = False
        config['test'] = True
        config['rl_device'] = 'cpu'
        config['sim_device'] = 'cpu'
        config['task']['env']['numEnvs'] = 1

    with launch_isaacgym_env(preprocess_func) as ret:
        cfg_dict = ret['config']
        env = ret['env']

        env = NumpyAloneEnv(env)

        if is_http_pub:
            env = HttpPubEnv(env)

        experiment_config = cfg_dict['task']['experiment']
        if 'rlgames' in experiment_config:
            executor = RlgamesPredictor.load_model(**experiment_config['rlgames'])
        elif 'http' in experiment_config:
            executor = HttpPredictor()
        else:
            raise NotImplementedError()

        runner = ExerimentRunner(cfg_dict, env, experiment, executor, epoch=experiment_config.get('epoch', 3))
        runner.start()