from math import sqrt
from random import randint, choice
from enum import Enum

import torch
import torchvision.transforms.functional as F
import numpy as np
from loguru import logger

from pumbaa.common.default import wheel_radius, wheel_points, flipper_length
from pumbaa.helper.pose import RobotRecommandPoseHelper

from processing.robot_info.robot_point import robot_to_world, robot_flipper_positions
from processing.robot_info.velocity import world_velocity_to_v_w
from pumbaa.utils.geo import point_in_rotated_ellipse
from utils.debug.run_time import log_mean_run_time

from .common_task import PumbaaCommonTask


class CrossingTask(PumbaaCommonTask):

    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        PumbaaCommonTask.__init__(self, name, sim_config, env, offset)

    def register_gym_func(self):
        super().register_gym_func()
        self.done_componets.update({
            'target': self._is_done_in_target,
            'out_of_range': self._is_done_out_of_range,
            'rollover': lambda i: torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 80),
            'timeout': lambda i: self.num_steps[i] >= self.max_step,
        })
        self.reward_func_deps.update({
            'yaw': 'target_directs',
        })
        self.reward_componet_register.update({
            'yaw': lambda i: -self.target_directs[i] ** 2,
            'roll_angle': lambda i: -self.orientations_3[i, 0] ** 2,
            'pitch_angle': lambda i: -self.orientations_3[i, 1] ** 2,
            'angular_velocity': lambda i: -torch.sum(self.velocities[i, 3:5] ** 2),
            'robot_height': lambda i: -torch.relu(self.positions[i][2] - wheel_radius - torch.max(self.current_frame_height_maps[i])),
        })

    def _calc_asymmetry(self, img):
        return torch.mean(torch.abs(img - F.hflip(img)))

    def _calculate_metrics_flipper_symmetry(self, i):
        asymmetry_coef = 100

        flippers = torch.deg2rad(self.flipper_positions[i])
        front_asymmetry = self._calc_asymmetry(self.current_frame_height_maps[i][self.height_map_size[0] // 2:, :])
        rear_asymmetry = self._calc_asymmetry(self.current_frame_height_maps[i][:self.height_map_size[0] // 2, :])
        # print('-----------------', flippers, front_asymmetry, rear_asymmetry)

        return -torch.relu(torch.abs(flippers[0] - flippers[1]) - asymmetry_coef * front_asymmetry) \
            - torch.relu(torch.abs(flippers[2] - flippers[3]) - asymmetry_coef * rear_asymmetry)

    def _calculate_metrics_z_acceleration(self, i):
        # positions = self.history_positions[i]
        #
        # if len(positions) < 3:
        #     return 0
        #
        # return positions[-1][2] - 2 * positions[-2][2] + positions[-3][2]

        velocitiy = self.velocities[i]

        if len(velocitiy) < 2:
            return 0

        return -(velocitiy[-1] - velocitiy[-2]) ** 2

    def _calculate_metrics_baselink_force(self, i):
        f = self.baselink_contacts[i].get_net_contact_forces()
        f_up = torch.sum(f[:, 2])
        return -1 if f_up == 0 else 0

    def _calculate_metrics_flipper_force(self, i):
        f = self.flipper_contacts[i].get_all_flipper_net_forces()
        f = torch.sign(torch.abs(f))

        return -(torch.abs(f[0, 2] - f[1, 2]) + torch.abs(f[2, 2] - f[3, 2]))

    def _calculate_metrics_angular_acceleration(self, i):
        velocities = self.history_velocities[i]

        if len(velocities) < 2:
            return 0

        return -torch.sum((velocities[-1] - velocities[-2])[3:5] ** 2)

    def _calculate_metrics_stick_rear_flipper(self, i):
        roll = self.orientations_3[i][0]
        pitch = self.orientations_3[i][1]

        _t = [flipper_length * s / 6 + wheel_radius for s in range(2, 7)]

        r = 0
        for k in [-1, -2]:
            flipper_points = torch.cat(
                [robot_flipper_positions(self.flipper_positions[i], degree=True, flipper_length=j)[k, :].view(1, 3) for j in _t]
            )
            flipper_points = robot_to_world(roll, pitch, flipper_points).to(self.device)
            flipper_map_point = self.pose_helper.get_point_height(self.current_frame_height_maps[i], flipper_points[:, :2])
            flipper_h = flipper_points[:, 2] + self.positions[i, 2] - wheel_radius - flipper_map_point
            r -= torch.min(torch.relu(flipper_h))

        return r

    def _calculate_metrics_stick_baselink(self, i):
        roll = self.orientations_3[i][0]
        pitch = self.orientations_3[i][1]

        # track
        track_points = torch.DoubleTensor(wheel_points['left_track'] + wheel_points['right_track'])
        track_points = torch.cat([track_points, torch.zeros((len(track_points), 1))], dim=-1)
        track_points = robot_to_world(roll, pitch, track_points).to(self.device)

        track_map_point = self.pose_helper.get_point_height(self.current_frame_height_maps[i], track_points[:, :2])

        track_h = track_points[:, 2] + self.positions[i, 2] - wheel_radius - track_map_point

        track_h = torch.relu(track_h)
        track_reward = torch.relu(3 - torch.sum(track_h < 1e-4)) * torch.max(track_h)

        return -track_reward


    def _calculate_metrics_flipper_range(self, i):
        font = torch.deg2rad(self.flipper_positions[i, :2])
        rear = torch.deg2rad(self.flipper_positions[i, 2:])

        return -((2 * (font[0] - font[1]) / torch.pi) ** 2 + (2 * (rear[0] - rear[1]) / torch.pi) ** 2)

    def _calculate_metrics_flipper_pos(self, i):
        flipper = torch.deg2rad(self.flipper_positions[i])
        pos = self.pose_helper.calc_flipper_position(self.current_frame_height_maps[i]).mean(dim=-1)

        recommand_pos = torch.deg2rad(pos) + self.orientations_3[i, 1] * torch.tensor([1, 1, -1, -1])

        return -torch.sum((2 * (flipper - recommand_pos) / torch.pi) ** 4)

    def _calculate_metrics_contact(self, i):

        baselink_points = self.baselink_contacts[i].get_contact_points()
        flipper_points = self.flipper_contacts[i].get_contact_points()

        if len(baselink_points) == 0:
            return -1

        if len(baselink_points) + len(flipper_points) < 3:
            return -1
        return 0

    def _calculate_metrics_shutdown(self, i, N=5):

        if len(self.history_positions[i]) < N:
            return 0

        trajectory = torch.stack(list(self.history_positions[i])[-N:], dim=0)
        d_list = (trajectory[:, :2] - self.start_positions[i][:2]).norm(dim=1)

        forward_d = torch.diff(d_list)

        if torch.sum(forward_d[forward_d >= 0]) <= 1e-4:
            return -1

        return 0

    def _calculate_metrics_forward(self, i):
        target = self.target_positions[i]

        if len(self.history_positions[i]) <= 2:
            return 0

        trajectory = torch.stack(list(self.history_positions[i]), dim=0)
        d_list = (trajectory[:, :2] - target[:2]).norm(dim=1)
        d_max = (self.start_positions[i][:2] - target[:2]).norm()
        forward_d = d_list[-2] - d_list[-1]

        return (torch.relu(forward_d / d_max) * 100) ** 2

    def _calculate_metrics_end(self, i, R=100):
        r_end = 0

        # 到目标区域奖励
        if self._is_done_in_target(i):
            r_end = R

        # 超出范围惩罚
        elif self._is_done_out_of_range(i):
            r_end = -R / 4

        # 翻车惩罚
        elif torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 80):
            r_end = -R / 2

        return r_end

    def _get_observations_target_distances(self):
        return (self.target_positions - self.positions).norm(dim=1) / (self.target_positions - self.start_positions).norm(dim=1)

    def _get_observations_target_directs(self):

        robot_angle = self.orientations_3[:, 2]

        p_t = (self.target_positions - self.positions)[:, :2]
        dot_product = p_t[:, 0]
        vec_len = p_t.norm(dim=-1)
        target_angle = torch.sign(self.target_positions[:, 0]) * torch.arccos(dot_product / vec_len)

        target_directs = (target_angle - robot_angle) % torch.pi
        index = (target_directs > torch.pi / 2)
        target_directs[index] -= torch.pi

        self.target_directs = target_directs

        return target_directs.view(-1, 1)

    def _is_done_in_target(self, index):
        point = self.positions[index][:2]
        target = self.target_positions[index][:2]

        return (point - target).norm() <= 0.25

    def _is_done_out_of_range(self, index):
        point = self.positions[index]
        center = (self.start_positions[index] + self.target_positions[index]) / 2

        op = self.target_positions[index] - self.start_positions[index]
        d_max = op[:2].norm()
        theta = torch.arctan(op[1] / op[0])

        return not point_in_rotated_ellipse(
            point[0], point[1],
            center[0], center[1],
            d_max + 0.2, d_max / 2 + 0.1,
            theta
        )

    def take_actions(self, actions, indices):
        ret = self._action_mode_execute.convert_actions_to_std_dict(actions, default_v=self.max_v, default_w=0)
        self.articulation_view.set_v_w(ret['vel'], indices=indices)
        self._flipper_control.set_pos_dt_with_max(ret['flipper'], 60, index=indices)
        self.articulation_view.set_all_flipper_position_targets(self._flipper_control.positions)
        # self.articulation_view.set_all_flipper_positions(self._flipper_control.positions)
        # print(actions)
        return ret['vel'], ret['flipper']

