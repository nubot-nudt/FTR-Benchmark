import torchvision.transforms.functional as F
from functools import partial

from ptask_envs.pumbaa.common.default import wheel_radius

from ptask_envs.tasks.utils.geo import point_in_rotated_ellipse, distance_to_line_2d

from .pumbaa_task import PumbaaCommonTask
from .utils.quat import *


def _calc_h_asymmetry(img):
    return torch.mean(torch.abs(img - F.hflip(img)))


def _calc_v_asymmetry(img):
    return torch.mean(torch.abs(img - F.vflip(img)))


class CrossingTask(PumbaaCommonTask):

    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        PumbaaCommonTask.__init__(self, name, sim_config, env, offset)
        self.pitch_threshold = sim_config.config['task']['task']['constant']['pitch_threshold']

    def register_gym_func(self):
        super().register_gym_func()
        self.done_components.update({
            'target': self._is_done_in_target,
            'out_of_range': self._is_done_out_of_range,
            'rollover': lambda i: torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 80),
            'timeout': lambda i: self.num_steps[i] >= self.max_step,
            'deviation': lambda i: 0.5 - distance_to_line_2d(self.start_positions[i], self.target_positions[i],
                                                             self.positions[i]) < 0,
            'overspeed': self._is_done_overspeed,
        })
        self.reward_func_deps.update({
            'yaw': 'target_directs',
        })

        def _reward_acc(env_ids, indics):
            velocities = self.history_velocities[env_ids]
            if len(velocities) < 2:
                return 0
            return -(velocities[-1][indics] - velocities[-2][indics]) ** 2

        self.reward_component_register.update({
            'cross': lambda i: torch.exp(-4 * (self.velocity_commands[i, 0] - self.velocities[i, 0]) ** 2)
                               * self.current_frame_height_maps[i].std()
                               * 5 * (0.2 - distance_to_line_2d(self.start_positions[i], self.target_positions[i],
                                                                self.positions[i])),

            'yaw': lambda i: -self.target_directs[i] ** 2 * (
                    (self.positions[i] - self.target_positions[i])[:2].norm() > 0.5),

            'deviation': lambda i: -torch.relu(
                distance_to_line_2d(self.start_positions[i], self.target_positions[i], self.positions[i]) - 0.1),

            'roll_angle': lambda: -self.orientations_3[:, 0] ** 2,
            'pitch_angle': lambda: -self.orientations_3[:, 1] ** 2,
            'excess_pitch': lambda: -torch.relu(self.orientations_3[:, 1].abs() - torch.pi * self.pitch_threshold / 180) ** 2,

            'robot_height': lambda: -torch.relu(
                self.positions[:, 2] - wheel_radius -
                self.current_frame_height_maps.view(self.num_envs, -1).max(dim=-1)[0]),

            'lin_vel_x': lambda: -self.velocities[:, 0] ** 2,
            'lin_vel_y': lambda: -self.velocities[:, 1] ** 2,
            'lin_vel_z': lambda: -self.velocities[:, 2] ** 2,

            'ang_vel_xy': lambda: -self.velocities[:, 5] ** 2,
            'ang_vel_xz': lambda: -self.velocities[:, 4] ** 2,
            'ang_vel_yz': lambda: -self.velocities[:, 3] ** 2,

            'lin_acc_x': partial(_reward_acc, indics=0),
            'lin_acc_y': partial(_reward_acc, indics=1),
            'lin_acc_z': partial(_reward_acc, indics=2),

            'ang_acc_xy': partial(_reward_acc, indics=5),
            'ang_acc_xz': partial(_reward_acc, indics=4),
            'ang_acc_yz': partial(_reward_acc, indics=3),

            'chassis_fall': lambda: -1 * (self.chassis_forces[:, :, :, 2].view(self.num_envs, -1) > 8).sum(-1),
        })

    def _calculate_metrics_flipper_symmetry(self, i):
        flippers = self.flipper_positions[i]
        coef = 24

        front_asymmetry = _calc_h_asymmetry(self.current_frame_height_maps[i][self.height_map_size[0] // 2:, :])
        rear_asymmetry = _calc_h_asymmetry(self.current_frame_height_maps[i][:self.height_map_size[0] // 2, :])
        left_asymmetry = _calc_v_asymmetry(self.current_frame_height_maps[i][:, self.height_map_size[1] // 2:])
        right_asymmetry = _calc_v_asymmetry(self.current_frame_height_maps[i][:, :self.height_map_size[1] // 2])

        return - torch.exp(-coef * front_asymmetry ** 2) * (flippers[0] - flippers[1]) ** 2 \
            - torch.exp(-coef * rear_asymmetry ** 2) * (flippers[2] - flippers[3]) ** 2 \
            - torch.exp(-coef * left_asymmetry ** 2) * (flippers[0] - flippers[2]) ** 2 \
            - torch.exp(-coef * right_asymmetry ** 2) * (flippers[1] - flippers[3]) ** 2

    def _calculate_metrics_flipper_height(self, i):
        flipper_points = self.flipper_world_pos[i][[0, 1, 2, 3], [-1, -1, 0, 0], :]

        r = 0
        for p in flipper_points:
            h = self.hmap_convertor.get_world_point_height(p)
            flipper_h = p[2] - wheel_radius - h
            r -= torch.relu(flipper_h)
        return r

    # def _calculate_metrics_stick_baselink(self, i):
    #
    #     # track
    #     track_points = torch.FloatTensor(wheel_points['left_track'] + wheel_points['right_track'])
    #     track_points = torch.cat([track_points, torch.zeros((len(track_points), 1))], dim=-1)
    #     # track_points = robot_to_world(roll, pitch, track_points).to(self.device)
    #     track_points = quat_apply_pitch_and_roll(self.orientations[i], track_points)
    #
    #     track_map_point = self.pose_helper.get_point_height(self.current_frame_height_maps[i], track_points[:, :2])
    #
    #     track_h = track_points[:, 2] + self.positions[i, 2] - wheel_radius - track_map_point
    #
    #     track_h = torch.relu(track_h)
    #     track_reward = torch.relu(3 - torch.sum(track_h < 1e-4)) * torch.max(track_h)
    #
    #     return -track_reward

    # def _calculate_metrics_flipper_pos(self, i):
    #     flipper = self.flipper_positions[i]
    #     pos = self.pose_helper.calc_flipper_position(self.current_frame_height_maps[i]).mean(dim=-1)
    #
    #     recommand_pos = pos + self.orientations_3[i, 1] * torch.tensor([1, 1, -1, -1])
    #
    #     return -torch.sum((2 * (flipper - recommand_pos) / torch.pi) ** 4)

    def _calculate_metrics_forward(self, i):
        target = self.target_positions[i]

        if len(self.history_positions[i]) <= 2:
            return 0

        trajectory = torch.stack(list(self.history_positions[i]), dim=0)
        d_list = (trajectory[:, :2] - target[:2]).norm(dim=1)
        d_max = (self.start_positions[i][:2] - target[:2]).norm()
        forward_d = d_list[-2] - d_list[-1]

        return forward_d / d_max

    def _calculate_metrics_shutdown(self, i):

        N = 3

        if len(self.history_positions[i]) < N:
            return 0

        trajectory = torch.stack(list(self.history_positions[i])[-N:], dim=0)
        d_list = (trajectory[:, :2] - self.start_positions[i][:2]).norm(dim=1)

        forward_d = torch.diff(d_list)

        if torch.sum(forward_d[forward_d >= 0]) <= 1e-3:
            return -1

        return 0

    def _calculate_metrics_end(self, i):
        R = 10
        r_end = 0

        # 到目标区域奖励
        if self._is_done_in_target(i):
            r_end = R

        # 超出范围惩罚
        # elif self._is_done_out_of_range(i):
        #     r_end = -R / 4

        # 翻车惩罚
        elif torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 80):
            r_end = -R / 2

        # # 超速
        # elif self._is_done_overspeed(i):
        #     r_end = -R / 2

        return r_end

    def _get_observations_robot_linear_vels(self):
        return self.velocities[:, :3]

    def _get_observations_robot_angular_vels(self):
        return self.velocities[:, 3:]

    def _get_observations_target_distances(self):
        return (self.target_positions - self.positions).norm(dim=1) / (
                self.target_positions - self.start_positions).norm(dim=1)

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

    def _is_done_overspeed(self, i):
        if torch.any(self.velocities[i, :3] > 50):
            return True

        if torch.any(self.velocities[i, 3:] > torch.pi * 4):
            return True

        return False

    def _is_done_out_of_range(self, index):
        point = self.positions[index]
        center = (self.start_positions[index] + self.target_positions[index]) / 2

        op = self.target_positions[index] - self.start_positions[index]
        d_max = op[:2].norm()
        theta = torch.arctan(op[1] / op[0])

        return not point_in_rotated_ellipse(
            point[0], point[1],
            center[0], center[1],
            d_max / 2 + 0.2, d_max / 4 + 0.1,
            theta
        )

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self.default_v[env_ids] = (torch.rand((len(env_ids),), device=self.device)) * (
                self.max_v - self.min_v) + self.min_v
        # self.default_v[env_ids] = 0.2

    def take_actions(self, actions, indices):
        # print(actions)

        # if torch.any(actions < -0.9):
        #     print(actions[actions < -0.9])

        ret = self._action_mode_execute.convert_actions_to_std_dict(actions, default_v=self.default_v, default_w=0)
        self.pumbaa_robots.set_v_w(ret['vel'], indices=indices)

        # if torch.any(ret['flipper'] < -1.8):
        #     print(ret['flipper'][ret['flipper'] < -1.8])

        self._flipper_control.set_pos_dt_with_max(ret['flipper'], self.flipper_pos_max, index=indices)
        self.pumbaa_robots.set_all_flipper_position_targets(self._flipper_control.positions, indices=indices,
                                                            degree=True)

        # if torch.any(self._flipper_control.positions < -30):
        #     print(self._flipper_control.positions[self._flipper_control.positions < -30])

        # for i in range(self.num_envs):
        #     pos = self.pose_helper.calc_flipper_position(self.current_frame_height_maps[i]).mean(dim=-1)
        #     recommand_pos = pos + self.orientations_3[i, 1] * torch.tensor([1, 1, -1, -1])
        #     self.pumbaa_robots.set_all_flipper_position_targets(recommand_pos.to(torch.float32))

        return ret['vel'], ret['flipper']

    def cleanup(self) -> None:
        super().cleanup()
        self.default_v = torch.zeros((self.num_envs,), device=self.device)
