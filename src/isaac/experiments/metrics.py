

import pandas as pd
from experiments.base import Experiment

class MetricsExperiment(Experiment):
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
        df.to_csv('./data/experiment/metrics')

    def before_episode(self, epoch_num):
        super().before_episode(epoch_num)
        self.episode_start_time = self.task.current_time

    def after_episode(self, epoch_num):
        super().after_episode(epoch_num)

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)

        position = self.task.positions
        orientation = self.task.orientations_3
        velocity = self.task.velocities
        sim_time = self.task.current_time - self.episode_start_time

        x, y, z = position[0].tolist()
        roll, pitch, yaw = orientation[0].tolist()

        lin_x, lin_y, lin_z, ang_roll, ang_pitch, ang_yaw = velocity[0].tolist()

        self.result.append({
            'epoch': self.epoch_num,
            'step': self.step_num,

            'pitch': pitch,
            'roll': roll,
            'yaw': yaw,

            'x': x,
            'y': y,
            'z': z,

            'lin_x': lin_x,
            'lin_y': lin_y,
            'lin_z': lin_z,

            'ang_roll': lin_z,
            'ang_pitch': lin_z,
            'ang_yaw': lin_z,

            'time': sim_time,
        })

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)