import pandas as pd
from utils.tensor import to_numpy
from experiments.base import Experiment

class ValueExperiment(Experiment):

    def before_init(self, cfg_dict, env):
        super().before_init(cfg_dict, env)

    def after_init(self, runner):
        super().after_init(runner)
        self.gamma = self.executor.model_loader.params['config']['gamma']

    def on_start(self):
        super().on_start()
        self.result = []
    def on_end(self):
        super().on_end()
        df = pd.DataFrame(self.result)
        df.to_csv('./data/experiment/value')

    def before_episode(self, epoch_num):
        super().before_episode(epoch_num)
        self.q_values = []
        self.state_values = []
        self.rewards = []

    def after_episode(self, epoch_num):
        super().after_episode(epoch_num)
        monte_values = [0] * len(self.rewards)
        monte_values[-1] = self.rewards[-1]
        for i in range(len(self.rewards) - 2, -1, -1):
            monte_values[i] = self.rewards[i] + self.gamma * monte_values[i + 1]

        result = []
        for i in range(0, len(self.state_values)):
            result.append({
                'index': self.epoch_num,
                'step': i,
                'monte_value': monte_values[i],
                'q_value': self.q_values[i],
                'state_value': self.state_values[i],
            })

        self.result += result

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)
        self.state_values.append(to_numpy(self.executor.get_value(obs).item()))

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)
        self.rewards.append(to_numpy(reward))
        self.q_values.append(self.rewards[-1] + self.gamma * to_numpy(self.executor.get_value(obs).item()))


class RewardExperiment(Experiment):

    def before_init(self, cfg_dict, env):
        super().before_init(cfg_dict, env)

    def after_init(self, runner):
        super().after_init(runner)

    def on_start(self):
        super().on_start()
        self.result = []

    def on_end(self):
        super().on_end()
        df = pd.DataFrame(self.result)
        df.to_csv('./data/experiment/reward')

    def before_episode(self, epoch_num):
        super().before_episode(epoch_num)
        self.reward_list = []

    def after_episode(self, epoch_num):
        super().after_episode(epoch_num)
        self.result += self.reward_list

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)

        reward_infos = self.env.get_reward_infos()

        for reward_info in reward_infos:
            info_ = {
                'epoch': self.epoch_num,
                'step': self.step_num,
                **reward_info,
            }
            info_['reward'] = to_numpy(info_['reward'])
            self.reward_list.append(info_)
