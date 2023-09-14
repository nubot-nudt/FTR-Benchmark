from math import sqrt
from random import randint, choice

import torch

import numpy as np
from loguru import logger
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

from ..common.default import wheel_radius
from ..helper.pose import RobotRecommandPoseHelper

from .rl import PumbaaBaseRLTask

class PumbaaTask(PumbaaBaseRLTask):

    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self.height_map_length = 112

        if not hasattr(self, 'observation_componets'):
            self.observation_componets = [
                # [ length, observation_func ]
                [self.height_map_length, self._get_observations_height_maps],
                [2, self._get_observations_robot_vels],
                [4, self._get_observations_robot_flippers],
                [2, self._get_observations_robot_orients],
                [1, self._get_observations_target_directs],
                [3, self._get_observations_distances],

                # [4, self._get_observations_recommend_flipper_positions],
            ]

        self.reward_componets = [
            self._calculate_metrics_end,
            self._calculate_metrics_distance,
            self._calculate_metrics_yaw,
            self._calculate_metrics_flipper_pos,
            self._calculate_metrics_flipper_range,
            self._calculate_metrics_stable,
            self._calculate_metrics_contact,
        ]

        self._num_observations = sum(i[0] for i in self.observation_componets)
        self._num_states = self._num_observations
        PumbaaBaseRLTask.__init__(self, name, sim_config, env, offset)

        if len(self._reward_coef) != len(self.reward_componets):
            logger.warning('_reward_coef is not match reward_componets')

            if len(self.reward_componets) > len(self._reward_coef):
                self._reward_coef += [0] * (len(self.reward_componets) - len(self._reward_coef))

        self.extractor = torch.nn.AvgPool2d(3)
        self.pose_helper = RobotRecommandPoseHelper(shape=(16, 7), scale=(2.5, 1.2))

    def _calculate_metrics_flipper_range(self, i, max_diff_angle=45):
        font = self.flipper_positions[i, :2]
        rear = self.flipper_positions[i, 2:]

        r = 0

        if torch.abs(font[0] - font[1]) > max_diff_angle:
            r -= 1
        if torch.abs(rear[0] - rear[1]) > max_diff_angle:
            r -= 1
        return 0.5 * r

    def _calculate_metrics_flipper_pos(self, i, tolerance=20):
        flipper = self.flipper_positions[i]
        recommand_pos = torch.rad2deg(self.recommend_flipper_positions[i, :])
        deleta_pos = torch.abs(flipper - recommand_pos)

        r = torch.zeros((4, ))

        index = (deleta_pos > tolerance)
        r[index] = torch.clip(deleta_pos[index] / (120 - tolerance * 2), 0, 1)

        return -0.25 * r.sum()

    def _calculate_metrics_yaw(self, i, tolerance=20):
        delta_angle = torch.rad2deg(self.target_directs[i])

        r = 0
        if torch.abs(delta_angle) > tolerance:
            r -= 1

        return r

    def _calculate_metrics_contact(self, i):

        baselink_left_forces, baselink_right_forces = self.baselink_contacts[i].get_left_and_right_forces()
        flipper_left_forces, flipper_right_forces = self.flipper_contacts[i].get_left_and_right_forces()

        # TODO contact reward

        baselink_points = self.baselink_contacts[i].get_contact_points()
        flipper_points = self.flipper_contacts[i].get_contact_points()

        if len(baselink_points) < 1:
            return -1

        if len(baselink_points) + len(flipper_points) < 3:
            return -1

        if len(baselink_points) == 2:
            return -0.5

        return 0


    def _calculate_metrics_stable(self, i, pitch_threshold=30, roll_threshold=30):
        # 越障平稳性辅助奖励
        if len(self.angles[i]) <= 1:
            r_stable = 0
        else:
            angle = torch.stack(self.angles[i], dim=0)
            pitch = torch.abs(angle[:, 1])
            roll = torch.abs(angle[:, 0])

            delta_pitch = torch.diff(pitch)
            delta_roll = torch.diff(roll)

            lambda_l = 1 / 3

            if pitch[-1] > pitch_threshold and delta_pitch[-1] > 0:
                r_stable_pitch = -1
            elif torch.mean(delta_pitch) > (1 / lambda_l):
                r_stable_pitch = -1
            else:
                r_stable_pitch = torch.clip(-lambda_l * torch.mean(torch.abs(torch.diff(angle[:, 1]))), -1, 1)

            if roll[-1] > roll_threshold and delta_roll[-1] > 0:
                r_stable_roll = -1
            elif torch.mean(delta_roll) > (1 / lambda_l):
                r_stable_roll = -1
            else:
                r_stable_roll = torch.clip(-lambda_l * torch.mean(torch.abs(torch.diff(angle[:, 0]))), -1, 1)

            r_stable = r_stable_pitch + r_stable_roll
        return r_stable

    def _calculate_metrics_distance(self, i):
        target = self.target_positions[i]

        if len(self.trajectories[i]) <= 1:
            return 0

        trajectory = torch.stack(self.trajectories[i], dim=0)
        beta_d = -(trajectory[:, :2] - target[:2]).norm(dim=1)
        d = (self.start_positions[i][:2] - target[:2]).norm()
        forward_d = (beta_d[-1] - beta_d[-2])
        r_distance = torch.clip(forward_d / d * 100, -1, 1)

        if len(beta_d) <= 1:
            r_distance = 0
        elif torch.std(beta_d) <= 1e-5:
            r_distance = -1
        elif forward_d < 0:
            r_distance = -1
        elif beta_d[-1] / d // 0.05 >= beta_d[-2] / d // 0.05:
            r_distance = 1

        return r_distance

    def _calculate_metrics_end(self, i, R=150):

        # 到目标区域奖励
        if self._is_done_in_target(i):
            r_end = R
        # 时间惩罚
        elif self.num_steps[i] >= self.max_step:
            r_end = -R
        # 偏航惩罚
        elif self._is_done_out_of_range(i):
            r_end = -R
        # 翻车惩罚
        elif torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 60):
            r_end = -R
        else:
            r_end = 0
        return r_end

    def _get_observations_recommend_flipper_positions(self):
        self.recommend_flipper_positions = torch.zeros((self.num_envs, 4), device=self.device)

        for i in range(self.num_envs):
            pos = self.pose_helper.calc_flipper_position(self.current_frame_height_maps[i]).mean(dim=-1)
            pitch = torch.rad2deg(self.orientations_3[i, 1])  # 向上为负

            # pitch = self.pose_helper.calc_robot_orient(self.current_frame_height_maps[i])[0]
            # pitch = torch.rad2deg(pitch)

            recommand_pos = pos + pitch * torch.tensor([1, 1, -1, -1])
            self.recommend_flipper_positions[:, ] = torch.deg2rad(recommand_pos)

        return self.recommend_flipper_positions

    def _get_observations_height_maps(self):
        buf = torch.zeros((self.num_envs, self.height_map_length), device=self.device)

        for i in range(self.num_envs):
            # quat_to_euler_angles(self.orientations[i], degrees=True)[2]
            local_map = self.map_helper.get_obs(self.positions[i].cpu(), torch.rad2deg(self.orientations_3[i]).numpy()[2], (2.5, 1.2))
            local_map = torch.from_numpy(local_map).to(self.device)
            if local_map.shape != (48, 22):
                continue
            if hasattr(self, 'extractor'):
                ext_map = self.extractor(torch.reshape(local_map, (1, 1, 48, 22)))
            else:
                ext_map = local_map

            ext_map -= self.positions[i][2] - wheel_radius

            buf[i, :] = ext_map.flatten()

        self.current_frame_height_maps = buf
        return buf

    def _get_observations_distances(self):

        target = self.target_positions
        point = self.positions
        d_max = (target - point).norm(dim=1)

        return (target - point) / d_max.view(-1, 1)

    def _get_observations_robot_orients(self):
        return self.orientations_3[:, :2]

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

    def _get_observations_robot_flippers(self):
        return self.flipper_positions / 180 * torch.pi

    def _get_observations_robot_vels(self):
        return torch.stack(self.robot_view.get_v_w(), dim=1)

    def get_observations(self) -> dict:

        index = 0
        for length, func in self.observation_componets:
            ret = func()

            # logger.debug(f'ret.shape[1] != length in {func.__name__}, shape is {ret.shape[0]} and length is {length}')
            assert ret.shape[1] == length

            self.obs_buf[:, index:index + length] = ret

            index += length

        return self.obs_buf

    def prepare_to_calculate_metrics(self):

        field_func_maps = {
            'recommend_flipper_positions': self._get_observations_recommend_flipper_positions,
        }

        for field, func in field_func_maps.items():
            if hasattr(self, field):
                continue
            func()


    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0

        self.prepare_to_calculate_metrics()

        for i in range(self.num_envs):
            self.rew_buf[i] = 0
            for weight, func in zip(self._reward_coef, self.reward_componets):
                value = func(i)
                self.rew_buf[i] += weight * value
                self._reward_value_dict[func.__name__].append(value)
