from abc import ABC
from itertools import cycle
from collections import deque
import numpy as np
import torch
from loguru import logger

from omni.isaac.core.utils.prims import find_matching_prim_paths
from omni.isaac.core.utils.prims import get_prim_at_path
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.stage import get_current_stage

from ptask_envs.pumbaa.utils.prim import add_usd, update_collision
from ptask_envs.pumbaa.common.config import PumbaaTaskConfig
from ptask_envs.pumbaa.view.rigid import *
from ptask_envs.pumbaa.view.robot import PumbaaRobot, PumbaaRobotView
from ptask_envs.pumbaa.common.default import *
from ptask_common.processing.action_mode import ActionModeFactory
from ptask_common.processing.control.flipper import FlipperControl
from ptask_common.utils.tensor import to_numpy, to_tensor
from ptask_common.utils.common.birth import load_birth_info

from .rl_base import AutoConfigRLTask


def auto_gen_birth_info(map_array, ct):
    a = int(1.6 / ct.cell_size)
    d = int(4.0 / ct.cell_size)
    flat_regions = []
    rows, cols = map_array.shape

    for i in range(rows - a + 1):
        for j in range(cols - a + 1):
            sub_map = map_array[i:i + a, j:j + a]
            if np.mean(sub_map) > -1 and np.std(sub_map) < 0.01:
                flat_regions.append((i + a // 2, j + a // 2))

    s_t_list = []
    for p in flat_regions:
        for direct in [(1, 0, 0), (-1, 0, -np.pi), (0, 1, np.pi / 2), (0, -np.pi / 2)]:
            p2 = [p[0] + direct[0] * d, p[1] + direct[1] * d]
            if p2[0] in range(rows) and p2[1] in range(cols):
                if map_array[p2[0], p2[1]] > -1:
                    s_t_list.append((p, p2, direct))

    reset_infos = []
    for s, t, d in s_t_list:
        s1 = ct.convert_array_index_to_world_point(s)
        t1 = ct.convert_array_index_to_world_point(t)

        orient = [0, 0, d[2]]
        # orient = quaternion_from_start_to_end(s1, t1)
        reset_infos.append({
            'start_point': [s1[0], s1[1], map_array[s[0], s[1]] + wheel_radius],
            'start_orient': orient,
            'target_point': [t1[0], t1[1], map_array[t[0], t[1]] + wheel_radius],
            'target_orient': orient,
        })
    return reset_infos


class PumbaaCommonTask(AutoConfigRLTask, ABC):
    _field_to_keys = {
        'max_episode_length',
        'state_space_dict',
        'action_mode',
        'reward_coef',
        'reset_info_maps_file',
    }

    def __init__(self, name, sim_config, env, offset=None) -> None:
        self.height_map_length = (2.25, 1.05)
        self.height_map_size = (45, 21)
        super().__init__(name, sim_config, env, offset)

        self.extractor = torch.nn.AvgPool2d(3)
        # self.pose_helper = RobotRecommandPoseHelper(shape=self.height_map_size, scale=self.height_map_length)

    def init_config(self, sim_config):
        super().init_config(sim_config)
        self.task_config = PumbaaTaskConfig(self._task_cfg)

    def register_gym_func(self):
        super().register_gym_func()

        self.observation_component_register.update({
            'cmd_vels': lambda: self.velocity_commands,
            'cmd_flippers': lambda: torch.rad2deg(self.flipper_positions) / self.flipper_dt,
            'robot_flippers': lambda: self.flipper_positions,
            'robot_vels': lambda: self.velocities[:, [0, -1]],
            'robot_lin_x': lambda: self.velocities[:, 0],
            'robot_orients': lambda: self.orientations_3[:, :2],
            'projected_gravity': lambda: self.projected_gravity,
        })

        self.reward_component_register.update({
            'action_power': lambda i: -self.actions[i].abs().mean(),
        })

    def _calculate_metrics_action_rate(self, i):
        actions = self.history_actions[i]

        if len(actions) < 2:
            return 0

        return -torch.mean((actions[-1] - actions[-2]) ** 2)

    def _calculate_metrics_action_smoothness(self, i):
        actions = self.history_actions[i]

        if len(actions) < 3:
            return 0

        return -torch.mean((actions[-1] - 2 * actions[-2] + actions[-3]) ** 2)

    def prepare_reset_info(self):
        """
        用于家在机器人初始和结束形象，结果会放入self._reset_info中
        :return:
        """
        auto_gen_flag = False
        if not hasattr('self', '_reset_info'):
            if self.asset.has_task_info():
                self._reset_info = [self.asset.task_info]
            elif hasattr(self, '_reset_info_maps_file'):
                try:
                    self._reset_info = load_birth_info(os.path.basename(self._task_cfg["asset"]),
                                                       self._reset_info_maps_file)
                except KeyError:
                    logger.warning('No reset information found, attempting to generate automatically')

        if not hasattr(self, '_reset_info') or len(self._reset_info) == 0:
            self._reset_info = auto_gen_birth_info(self.hmap_helper.map, self.hmap_convertor)
            auto_gen_flag = True

        if len(self._reset_info) == 0:
            raise NotImplementedError('could not load reset_info_maps_file.')

        # 对数据进行格式统一化
        for info in self._reset_info:
            if len(info['start_orient']) == 3:
                info['start_orient'] = euler_angles_to_quat(to_numpy(info['start_orient']))

            for key, value in info.items():
                info[key] = to_tensor(value)

        _data = self._reset_info
        _seq = cycle(self._reset_info)

        reset_choice_func = {
            'random': lambda: random.choice(_data),
            'sequence': lambda: next(_seq),
        }
        if auto_gen_flag:
            self._reset_info_generate = reset_choice_func['random']
        else:
            self._reset_info_generate = reset_choice_func[self.task_config.reset_type]

    @property
    def flipper_pos_max(self):
        return self.task_config.flipper_pos_max

    def _reset_idx_robot_info(self, num_resets):
        pos = torch.zeros((num_resets, 3), device=self._device)
        orient = torch.zeros((num_resets, 4), device=self._device)
        target = torch.zeros((num_resets, 3), device=self._device)
        if 'sync' in self.action_mode:
            flipper = torch.rand((num_resets, 2, 1), device=self._device).repeat((1, 1, 2)).view(-1, 4)
        else:
            flipper = torch.rand((num_resets, 4), device=self._device)
        flipper = flipper * self.flipper_pos_max

        for i in range(num_resets):
            _t = self._reset_info_generate()
            pos[i] = _t['start_point']
            orient[i] = _t['start_orient']
            target[i] = _t['target_point']

        return {
            'pos': pos,
            'orient': orient,
            'target': target,
            'flipper': flipper,
        }

    def _get_observations_height_2d_points(self):
        height_map = self._get_observations_height_maps().view(self.num_envs, 15, 7)
        return height_map.mean(dim=-1)

    def _get_observations_height_maps(self):
        buf = -torch.ones((self.num_envs, self.task_config.state_space_dict.get('height_maps', 105)),
                          device=self.device)

        for i in range(self.num_envs):
            local_map = self.hmap_helper.get_obs(self.positions[i].cpu(),
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

    def post_physics_step(self):
        # 计算 obs、reward、done 需要的数据
        self.positions, self.orientations = self.pumbaa_robots.get_world_poses()
        self.flipper_positions = self.pumbaa_robots.get_all_flipper_positions()
        world_vels = self.pumbaa_robots.get_velocities()
        self.velocities[:, :3] = quat_rotate_inverse(self.orientations, world_vels[:, :3])
        self.velocities[:, 3:] = quat_rotate_inverse(self.orientations, world_vels[:, 3:])
        self.projected_gravity = quat_rotate(self.orientations,
                                             torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(
                                                 (self._num_envs, 1)))
        self.chassis_forces = self.pumbaa_robots.chassis_wheel.get_net_contact_forces().view(self.num_envs, 2, 8, 3)
        self.flipper_forces = self.pumbaa_robots.flipper_wheel.get_net_contact_forces().view(self.num_envs, 4, 5, 3)
        self.flipper_world_pos = self.pumbaa_robots.flipper_wheel.get_world_poses()[0].view(self.num_envs, 4, 5, 3)

        self.orientations_3 = torch.stack(
            list(
                torch.from_numpy(quat_to_euler_angles(i)).to(self.device) for i in self.orientations.cpu()
            )
        )

        # 记录历史信息
        for i in range(self.num_envs):
            self.history_positions[i].append(self.positions[i])
            self.history_orientations_3[i].append(torch.rad2deg(self.orientations_3[i]))
            self.history_flippers[i].append(self.flipper_positions[i])
            self.history_actions[i].append(self.actions[i])
            self.history_velocities[i].append(torch.clone(self.velocities[i]))

        return super().post_physics_step()

    def pre_physics_step(self, actions) -> None:
        super().pre_physics_step(actions)
        if self.task_config.is_follow_camera:
            self.set_camera_follow(self.positions[0])

    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)

        if self.is_debug:
            logger.debug(f'reset indices={indices}')

        reset_info = self._reset_idx_robot_info(num_resets)

        self.pumbaa_robots.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
        self.pumbaa_robots.set_world_poses(reset_info['pos'], reset_info['orient'], indices=indices)

        self._flipper_control.set_pos(reset_info['flipper'], index=indices)
        self.pumbaa_robots.set_all_flipper_positions(reset_info['flipper'], indices=indices, degree=True)
        self.pumbaa_robots.set_v_w(torch.zeros((num_resets, 2), device=self._device), indices=indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.num_steps[env_ids] = 0

        # 重置计算所需要的变量
        for i in env_ids:
            self.history_positions[i].clear()
            self.history_orientations_3[i].clear()
            self.history_flippers[i].clear()
            self.history_actions[i].clear()
            self.history_velocities[i].clear()

        self.start_positions[env_ids] = reset_info['pos']
        self.start_orientations[env_ids] = reset_info['orient']
        self.target_positions[env_ids] = reset_info['target']

        self.post_reset_idx(env_ids)

    def set_action_mode(self, action_mode=None):
        if action_mode is None:
            action_mode = self.action_mode
        else:
            self._action_mode = action_mode

        self._action_mode_execute = ActionModeFactory.get_action_mode_by_name(
            action_mode,
            max_v=self.max_v,
            max_w=self.max_w,
            flipper_dt=self.flipper_dt,
            flipper_pos_max=self.flipper_pos_max,
        )

    def init_gym(self):
        super().init_gym()

        # action
        if not hasattr(self, 'max_v'):
            self.max_v = self.task_config.max_v

        if not hasattr(self, 'min_v'):
            self.min_v = self.task_config.min_v

        assert self.max_v >= self.min_v

        if not hasattr(self, 'max_w'):
            self.max_w = self.task_config.max_w

        if not hasattr(self, 'flipper_dt'):
            self.flipper_dt = self.task_config.flipper_dt

        self.set_action_mode()
        gym_info = self._action_mode_execute.gym_info

        if not hasattr(self, '_num_actions'):
            self._num_actions = gym_info['num_actions']
            self.action_space = gym_info['space']

    def post_reset(self):
        if not hasattr(self, '_init_finished_flag'):
            self.pumbaa_robots.find_idx()

            # for contact in chain(self.baselink_contacts, self.flipper_contacts):
            #     contact.initialize()

            if hasattr(self, 'imus'):
                for imu in self.imus:
                    imu.initialize()

            if self.device == 'cpu':
                for name, usd in self.obstacles.items():
                    update_collision(usd)

            self._init_finished_flag = True

    def set_up_scene(self, scene: Scene, apply_imu=False) -> None:
        self._stage = get_current_stage()
        # 添加第0号克隆体
        add_usd(self.pumbaa_usd, self.default_zero_env_path + '/' + self.pumbaa_usd.name)
        self.pumbaa_robot = PumbaaRobot(f"{self.default_zero_env_path}/{self.pumbaa_usd.name}/{robot_base_name}",
                                        name=robot_base_name)
        self._sim_config.apply_articulation_settings(
            robot_base_name, get_prim_at_path(self.pumbaa_robot.prim_path),
            self._sim_config.parse_actor_config(robot_base_name)
        )
        self.pumbaa_robot.set_pumbaa_default_properties(self._stage)
        self.pumbaa_robot.prepare_contacts(self._stage)
        self.pumbaa_robot.set_robot_env_config(self.task_config.pumbaa_cfg)
        super().set_up_scene(scene)

        # 添加机器人所需要用到的view
        self.pumbaa_robots = PumbaaRobotView(f'/World/envs/.*/{self.pumbaa_usd.name}/{robot_base_name}',
                                             track_contact_forces=True,
                                             prepare_contact_sensors=False)
        scene.add(self.pumbaa_robots)
        scene.add(self.pumbaa_robots.chassis_wheel)
        scene.add(self.pumbaa_robots.flipper_wheel)

        # 添加接触点传感器
        self.robot_prim_paths = find_matching_prim_paths(f'/World/envs/.*/{self.pumbaa_usd.name}')
        # print(self.robot_prim_paths)
        # if apply_imu:
        #     from pumbaa.sensor.imu import IMU
        #     self.imus = [IMU(path, scene) for path in self.robot_prim_paths]
        #
        # self.baselink_contacts = []
        # self.flipper_contacts = []
        # for path in self.robot_prim_paths:
        #     baselink_contact = PumbaaBaselinkContact(path, self._collision_filter_global_paths)
        #     flipper_contact = PumbaaAllFipperContact(path, self._collision_filter_global_paths)
        #     self.baselink_contacts.append(baselink_contact)
        #     self.flipper_contacts.append(flipper_contact)

        if self.asset.has_camera():
            self.set_initial_camera_params(
                camera_position=self.asset.camera['position'],
                camera_target=self.asset.camera['target'],
            )

        if self.device != 'cpu':
            for name, usd in self.obstacles.items():
                update_collision(usd, is_to_convex=True)

    @property
    def action_mode(self):
        return self._action_mode

    def init_task(self):
        super().init_task()
        self.pumbaa_usd = self.asset.robot
        self._flipper_control = FlipperControl(self.num_envs, device=self._device)

    def cleanup(self) -> None:
        super().cleanup()

        self.start_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.start_orientations = torch.zeros((self.num_envs, 4), device=self._device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.flipper_positions = torch.zeros((self.num_envs, 4), device=self._device)
        self.orientations = torch.zeros((self.num_envs, 4), device=self._device)
        self.orientations_3 = torch.zeros((self.num_envs, 3), device=self._device)
        self.velocities = torch.zeros((self.num_envs, 6), device=self._device)
        self.velocity_commands = torch.zeros((self.num_envs, 2), device=self._device)
        self.flipper_commands = torch.zeros((self.num_envs, 4), device=self._device)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=self.device)
        self.chassis_forces = torch.zeros((self.num_envs, 2, 8, 3), device=self.device)
        self.flipper_forces = torch.zeros((self.num_envs, 4, 5, 3), device=self.device)
        self.flipper_world_pos = torch.zeros((self.num_envs, 4, 5, 3), device=self.device)

        N = 5
        self.history_positions = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_orientations_3 = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_flippers = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_actions = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_velocities = [deque(maxlen=N) for _ in range(self.num_envs)]

        self.current_frame_height_maps = torch.zeros((self.num_envs, *self.height_map_size), device=self.device)
