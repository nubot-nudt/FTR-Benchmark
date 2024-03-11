from math import sqrt
from random import randint, choice
from collections import deque
from itertools import product

import torch

import numpy as np
from loguru import logger

from pumbaa.common.default import wheel_radius, wheel_points, flipper_length
from pumbaa.helper.pose import RobotRecommandPoseHelper

from processing.robot_info.robot_point import robot_to_world, robot_flipper_positions
from processing.robot_info.velocity import world_velocity_to_v_w
from utils.debug.run_time import log_mean_run_time

from .rl_base import PumbaaBaseRLTask


class PumbaaCommonTask(PumbaaBaseRLTask):

    def __init__(self, name, sim_config, env, offset=None) -> None:
        super().__init__(name, sim_config, env, offset)

        self.height_map_length = (2.25, 1.05)
        self.height_map_size = (45, 21)

        self.extractor = torch.nn.AvgPool2d(3)
        self.pose_helper = RobotRecommandPoseHelper(shape=self.height_map_size, scale=self.height_map_length)

    def register_gym_func(self):
        super().register_gym_func()

        self.observation_componet_register.update({
            'cmd_flippers': lambda: self.flipper_positions / self.flipper_dt,
            'robot_flippers': lambda: torch.deg2rad(self.flipper_positions),
            'robot_vels': lambda: world_velocity_to_v_w(self.velocities),
            'robot_orients': lambda: self.orientations_3[:, :2],
        })

        self.reward_componet_register.update({

        })

    def _calculate_metrics_action_rate(self, i):
        actions = self.history_actions[i]

        if len(actions) < 2:
            return 0

        return -torch.sum((actions[-1] - actions[-2]) ** 2)

    def _calculate_metrics_action_smoothness(self, i):
        actions = self.history_actions[i]

        if len(actions) < 3:
            return 0

        return -torch.sum((actions[-1] - 2 * actions[-2] + actions[-3]) ** 2)

    def _get_observations_cmd_vels(self):
        if not hasattr(self, '_cmd_vels_coef'):
            self._cmd_vels_coef = torch.tensor([self.max_v, self.max_w], device=self.device)
        return self.velocity_commands / self._cmd_vels_coef

    def _get_observations_height_maps(self):

        buf = -torch.ones((self.num_envs, self.task_config.state_space_dict['height_maps']), device=self.device)

        if not hasattr(self, 'current_frame_height_maps'):
            self.current_frame_height_maps = torch.zeros((self.num_envs, *self.height_map_size), device=self.device)

        for i in range(self.num_envs):
            local_map = self.map_helper.get_obs(self.positions[i].cpu(),
                                                         torch.rad2deg(self.orientations_3[i]).cpu().numpy()[2],
                                                         self.height_map_length)
            if local_map is None:
                continue

            if local_map.shape != self.height_map_size:
                logger.error("Your map doesn't seem big enough.")
                continue

            local_map = torch.from_numpy(local_map).to(self.device)
            self.current_frame_height_maps[i, :, :] = local_map

            ext_map = self.extractor(torch.reshape(local_map, (1, 1, *self.height_map_size)))

            ext_map -= self.positions[i][2] - wheel_radius

            buf[i, :] = ext_map.flatten()

        # self.current_frame_height_maps = buf
        return buf


    def get_observations(self):

        index = 0
        for name, d in self.observation_componets.items():
            length = d['size']

            ret = d['func']()

            if len(ret.shape) != 2:
                try:
                    ret = ret.view(-1, length)
                except Exception as e:
                    raise RuntimeError(f'{name} could not reshape, return {ret.shape} but except {(-1, length)}')

            if ret.shape[1] != length:
                raise RuntimeError(f'{name} shape is not equal length, return {ret.shape[1]} but except {length}')

            if 'scale' in d:
                ret = ret / d['scale']
            if 'clip' in d:
                ret = torch.clip(ret, d['clip'][0], d['clip'][1])

            self.obs_buf[:, index:index + length] = ret

            index += length

        return self.obs_buf

    def prepare_to_calculate_metrics(self):
        if not hasattr(self, '_reward_func_deps_list'):
            self._reward_func_deps_list = []
            for reward_name, obs_name in self.reward_func_deps.items():

                if reward_name not in self.reward_componets:
                    continue
                if obs_name in self.observation_componets:
                    continue

                self._reward_func_deps_list.append(self.observation_componets[obs_name]['func'])

        for func in self._reward_func_deps_list:
            func()

    def _compute_reward(self, env_i, rew_i, rew_name, rew_coef, rew_func=None):

        if rew_func is None:
            rew_func = self.reward_componets[rew_name]['func']

        rew_value = rew_func(env_i)
        rew_weight_value = rew_value * rew_coef

        return rew_value, rew_weight_value


    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0

        self.prepare_to_calculate_metrics()
        
        if not hasattr(self, '_compute_reward_params'):
            # (env_i, rew_i, rew_name, rew_coef, rew_func)
            self._compute_reward_params = [ 
                (env_i, r_comp[0], r_comp[1][0], r_comp[1][1]['coef'], r_comp[1][1]['func']) 
                for env_i, r_comp in product(range(self.num_envs), enumerate(self.reward_componets.items()))
            ]

        self.reward_infos = []

        for env_i, rew_i, rew_name, rew_coef, rew_func in self._compute_reward_params:
            try:
                rew_value, rew_weight_value = self._compute_reward(env_i, rew_i, rew_name, rew_coef, rew_func)
            except Exception as e:
                logger.error(f'{rew_name} : {type(e)} {e}')
                raise e

            self.rew_buf[env_i] += rew_weight_value
            self.reward_vec_buf[env_i, rew_i] = rew_weight_value

            self._reward_value_dict[rew_name].append(rew_value)
            self.reward_infos.append({
                    'index': env_i,
                    'name': rew_name,
                    'reward': rew_value,
                    'coef': rew_coef,
            })

            if torch.abs(self.reward_vec_buf[env_i, rew_i]) > 1e5:
                logger.warning(f'{rew_name}={rew_weight_value} is abnormal value in calculate_metrics')


    def get_extras(self):

        if len(self._end_type_list) > 0:
            for end_type in [t for t in self.done_componets]:
                self.extras['end_type']['end_' + end_type] = self._end_type_list.count(end_type) / len(self._end_type_list)

        for name, value in self._reward_value_dict.items():
            self.extras['reward_componets'][name] = 0 if len(value) == 0 else sum(value) / len(value)


        return super().get_extras()
    
    def is_done(self) -> None:

        for i in range(self.num_envs):
            for t, f in self.done_componets.items():
                if f(i):
                    self.reset_buf[i] += 1
                    self._end_type_list.append(t)
                    if self.is_play or self.is_debug:
                        print(f'** {i}-th end with {t}')

        self.reset_buf = torch.clip(self.reset_buf, 0, 1)


    def cleanup(self) -> None:
        super().cleanup()
        self.reward_vec_buf = torch.zeros((self._num_envs, self.reward_dim), device=self._device, dtype=torch.float)

        self.extras['end_type'] = dict()
        self.extras['reward_componets'] = dict()

        self._end_type_list = deque(maxlen=self.num_envs * 2)
        self._reward_value_dict = {i: deque(maxlen=self.num_envs * 1) for i in self.reward_componets}

    @property
    def reward_dim(self):
        return len(self.reward_componets)