from datetime import datetime

import pandas as pd

from ptask_common.processing.robot_info.obs import ObsSlice
from ptask_common.utils.tensor import to_numpy

from ptask_envs.experiments.base import Experiment

class RobotStateExperiment(Experiment):
    def before_init(self, cfg_dict, env):
        super().before_init(cfg_dict, env)
        self.obs_slice = ObsSlice(cfg_dict['task']['task']['state_space'])

    def after_init(self, runner):
        super().after_init(runner)

    def on_start(self):
        super().on_start()
        self.result = []

    def on_end(self):
        super().on_end()
        df = pd.DataFrame(self.result)
        df.to_csv('./data/experiment/robot_state')

    def before_episode(self, epoch_num):
        super().before_episode(epoch_num)

    def after_episode(self, epoch_num):
        super().after_episode(epoch_num)

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)
        orient = self.obs_slice.get_array_by_name(obs, 'robot_orients')
        pitch = orient[1]
        self.result.append({
            'epoch': self.epoch_num,
            'step': self.step_num,
            'pitch': to_numpy(pitch),
            'time': datetime.now(),
        })

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)