# -*- coding: utf-8 -*-
"""
====================================
@File Name ：record.py
@Time ： 2024/3/30 下午3:46
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

import torch
import pickle
from ptask_envs.experiments.base import Experiment


class RecordExperiment(Experiment):
    def __init__(self, save_path):
        self.save_path = save_path

    def before_init(self, cfg_dict, env):
        super().before_init(cfg_dict, env)
        self.extractor = torch.nn.AvgPool2d(3)
        self.num_envs = cfg_dict['task']['env']['numEnvs']

    def after_init(self, runner):
        super().after_init(runner)

    def on_start(self):
        super().on_start()
        self.demos = []
        self.trj = [[] for _ in range(self.num_envs)]

    def on_end(self):
        super().on_end()
        print(f'demos.length={len(self.demos)}')
        with open(self.save_path, 'wb') as f:
            pickle.dump(self.demos, f)

    def before_step(self, step_num, obs, action):
        super().before_step(step_num, obs, action)
        for i in range(self.num_envs):
            hmap = self.task.current_frame_height_maps
            position = self.task.positions
            orientation = self.task.orientations_3
            velocity = self.task.velocities
            flippers = self.task.flipper_positions

            self.trj[i].append({
                'map': self.extractor(torch.reshape(hmap[i, :], (1, 1, *self.task.height_map_size))).view(15, 7),
                'pos': position[i],
                'orient': orientation[i],
                'v': velocity[i, 0],
                'w': velocity[i, -1],
                'flipper': flippers[i],
            })

    def after_step(self, step_num, obs, reward, done, info):
        super().after_step(step_num, obs, reward, done, info)
        for i in range(self.num_envs):
            if self.task._is_done_in_target(i):
                self.demos.append(self.trj[i])
                self.trj[i] = []





