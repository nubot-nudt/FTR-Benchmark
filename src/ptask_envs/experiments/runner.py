import os
import torch

from ptask_common.autonomic.executor import ExecutorFactory
from ptask_common.autonomic.executor.rlgames import RlgamesPredictor
from ptask_common.autonomic.executor.http import HttpPredictor

from ptask_envs.envs.start import launch_isaacgym_env

from .base import Experiment


class ExerimentRunner():
    def __init__(self, cfg_dict, env, experiment: Experiment, executor=None, epoch=10):
        self.experiment = experiment
        self.experiment.before_init(cfg_dict, env)
        self.num_envs = cfg_dict['task']['env']['numEnvs']

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

        if self.num_envs == 1:
            for i in range(1, self.epoch + 1):
                self.experiment.before_episode(i)
                self.run_once(i)
                self.experiment.after_episode(i)
        else:
            self.experiment.before_episode(-1)
            self.run_once(-1)
            self.experiment.after_episode(-1)

        self.experiment.on_end()

    def run_once(self, epoch):

        obs = self.env.reset()
        step = 1
        all_reward = torch.zeros((self.num_envs,))
        done_count = 0

        while True:
            action = self.executor.get_action(obs)

            self.experiment.before_step(step, obs, action)
            obs, reward, done, info = self.env.step(action)
            self.experiment.after_step(step, obs, reward, done, info)
            all_reward += reward

            step += 1

            if self.num_envs == 1:
                if done[0]:
                    print(f'epoch: {epoch}, reward: {all_reward.item()}')
                    break
            else:
                idx = done.nonzero(as_tuple=False).squeeze(-1)

                if len(idx) > 0:
                    done_count += torch.sum(done)
                    print(f'idx: {idx.tolist()}, epoch: {done_count},  reward: {all_reward[idx].tolist()}')

                    if done_count >= self.epoch:
                        break


def create_runner(experiment: Experiment, is_http_pub=False, executor=None, num_envs=1):
    def preprocess_func(config):
        # config['headless'] = False
        config['test'] = True
        config['rl_device'] = 'cpu'
        config['sim_device'] = 'cpu'
        config['task']['env']['numEnvs'] = num_envs
        config['num_envs'] = num_envs

    with launch_isaacgym_env(preprocess_func) as ret:
        cfg_dict = ret['config']
        env = ret['env']

        # env = NumpyAloneEnv(env)

        if is_http_pub:
            from loguru import logger
            logger.warning('Not impl HttpPubEnv')
            # from ptask_envs.envs.interface.http_pub import HttpPubEnv
            # env = HttpPubEnv(env)

        experiment_config = cfg_dict['task']['experiment']
        executor_name = experiment_config['executor_name']
        if 'rlgames' == executor_name:
            executor = RlgamesPredictor.load_model(**experiment_config['rlgames'])
        elif 'http' == executor_name:
            executor = HttpPredictor()
        elif 'ftr' == executor_name:
            import ptask_ftr.executor
            executor = ExecutorFactory().build('ftr', **experiment_config['ftr'])
        else:
            raise NotImplementedError(executor_name)

        runner = ExerimentRunner(cfg_dict, env, experiment, executor, epoch=experiment_config.get('epoch', 3))
        runner.start()
