import os
import pickle
import random
import torch

import yaml
from enum import Enum
from collections import deque

from loguru import logger

from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import find_matching_prim_paths
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.world import World

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from ..common.asset import AssetEntry
from ..common.action_mode import ActionMode, ActionModeExecute
from ..helper.map import MapHelper
from ..helper.geometry import Geometryhelper
from ..utils.prim import add_usd, update_collision
from ..view.articulation import PumbaaArticulationView
from ..view.robot import PumbaaRobot, PumbaaBaseLinkView, PumbaaRobotView
from ..view.rigid import *
from ..utils.geo import point_in_rotated_ellipse


class EndType(Enum):
    target = 'target'
    out_of_range = 'out_of_range'
    timeout = 'timeout'
    rollover = 'rollover'


class PumbaaBaseRLTask(RLTask):

    _field_to_key_maps = {
        '_max_episode_length': 'episodeLength',
        '_action_mode': 'actionMode',
        '_reward_coef': 'rewardCoef',
        '_reset_info_maps_file': 'resetInfoMaps',
        '_is_record_extras_info': 'record_extras_info',
    }

    _default_values = {
        '_action_mode': 'continuous_std_6',
        '_is_record_extras_info': True,
    }

    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        print(self._sim_config.config)
        self.is_play = (self._sim_config.config['test'] == True)
        self.is_debug = (self._sim_config.config['debug'] == True)
        logger.info(f'start with play={self.is_play} debug={self.is_debug}')

        self.init_field()

        self.init_gym()

        RLTask.__init__(self, name, env)

        self.init_task()

    def _reset_idx_robot_info(self, num_resets):
        pos = torch.zeros((num_resets, 3), device=self._device)
        orient = torch.zeros((num_resets, 4), device=self._device)
        target = torch.zeros((num_resets, 3), device=self._device)

        if hasattr(self, '_reset_info_dict'):
            _data = self._reset_info_dict
            for i in range(num_resets):
                _t = random.choice(_data)
                pos[i] = torch.from_numpy(_t['start_point'])
                orient[i] = torch.from_numpy(euler_angles_to_quat(_t['start_orient']))
                target[i] = torch.from_numpy(_t['target_point'])

        else:
            logger.error('Unable to determine the initial state of the robot. Please implement _reset_idx_robot_info method.')
            raise NotImplementedError('_reset_idx_robot_info')

        return {
            'pos': pos,
            'orient': orient,
            'target': target,
        }

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


    def is_done(self) -> None:

        for i in range(self.num_envs):

            # 超出范围重置
            if self._is_done_out_of_range(i):
                self.reset_buf[i] += 1
                self._end_type_list.append(EndType.out_of_range)
                if self.is_play or self.is_debug:
                    logger.info(f'{i}-th end_out_of_range')

            # 到达目标区域重置
            elif self._is_done_in_target(i):
                self.reset_buf[i] += 1
                self._end_type_list.append(EndType.target)
                if self.is_play or self.is_debug:
                    logger.info(f'{i}-th end_target')


            # 翻车重置
            if torch.any(torch.abs(torch.rad2deg(self.orientations_3[i][:2])) >= 60):
                self.reset_buf[i] += 1
                self._end_type_list.append(EndType.rollover)
                if self.is_play or self.is_debug:
                    logger.info(f'{i}-th end_rollover')


            # 达到最大次数结束
            if self.num_steps[i] >= self.max_step:
                self.reset_buf[i] += 1
                self._end_type_list.append(EndType.timeout)
                if self.is_play or self.is_debug:
                    logger.info(f'{i}-th end_timeout')

        if self.is_play or self.is_debug:
            env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:

                logger.info(f'reward: {self._total_rewards[env_ids]}')
                for item in self._total_rewards[env_ids].tolist():
                    self._history_rewards.append(item)
                logger.info(f'history mean reward: {sum(self._history_rewards) / len(self._history_rewards)}, len={len(self._history_rewards)}')


    def get_extras(self):
        if self._is_record_extras_info == False:
            return None

        if len(self._end_type_list) > 0:
            for end_type in [EndType.out_of_range, EndType.rollover, EndType.target, EndType.timeout]:
                self.extras['end_type']['end_' + end_type.name] = self._end_type_list.count(end_type) / len(self._end_type_list)

        for name, value in self._reward_value_dict.items():
            self.extras['reward_componets']['reward' + name.replace('_calculate_metrics', '')] = sum(value) / len(value)

        return super().get_extras()

    def post_physics_step(self):
        # 计算 obs、reward、done 需要的数据
        self.positions, self.orientations = self.robot_view.get_world_poses()
        self.flipper_positions = self.articulation_view.get_all_flipper_positions()

        self.orientations_3 = torch.stack(
            list(
                torch.from_numpy(quat_to_euler_angles(i)).to(self.device) for i in self.orientations
            )
        )

        # 记录历史信息
        for i in range(self.num_envs):
            N = 5

            self.trajectories[i].append(self.positions[i])
            while len(self.trajectories[i]) > N:
                self.trajectories[i].pop(0)

            self.angles[i].append(torch.rad2deg(self.orientations_3[i]))
            while len(self.angles[i]) > N:
                self.angles[i].pop(0)

        obs_buf, rew_buf, reset_buf, extras = super().post_physics_step()

        self._total_rewards += rew_buf
        # print(self._total_rewards)

        reset_buf = torch.sign(reset_buf)

        return obs_buf, rew_buf, reset_buf, extras


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)

        if self.is_debug:
            logger.debug(f'reset indices={indices}')

        reset_info = self._reset_idx_robot_info(num_resets)

        self.robot_view.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
        self.robot_view.set_world_poses(reset_info['pos'], reset_info['orient'], indices=indices)

        self.articulation_view.set_all_flipper_positions(torch.zeros((num_resets, 4), device=self._device), indices=indices)
        self.articulation_view.set_v_w(torch.zeros((num_resets, 2), device=self._device), indices=indices)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self.num_steps[env_ids] = 0

        # 重置计算所需要的变量
        for i in env_ids:
            self.trajectories[i].clear()
            self.angles[i].clear()

        self.flipper_positions = torch.zeros((num_resets, 4), device=self._device)
        self.start_positions[env_ids] = reset_info['pos']
        self.target_positions[env_ids] = reset_info['target']

        self._total_rewards[env_ids] = 0
        self.post_reset_idx(env_ids)

    def post_reset_idx(self, env_ids):
        ...

    def pre_physics_step(self, actions) -> None:
        if actions is None:
            return

        if not self._env._world.is_playing():
            return

        self.num_steps += 1

        # 获取需要重置和不需要重置的机器人下标
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self.num_envs, dtype=torch.int32, device=self._device)

        ret = self._action_mode_execute.convert_actions_to_std_dict(actions, default_v=0.2, default_w=0)

        self.articulation_view.set_v_w(ret['vel'], indices=indices)
        self.articulation_view.set_all_flipper_dt(ret['flipper'], indices=indices)

    def post_reset(self):
        self.articulation_view.find_idx()

        for contact in chain(self.baselink_contacts, self.flipper_contacts):
            contact.initialize()

        if self.device == 'cpu':
            for name, usd in self.obstacles.items():
                update_collision(usd)

        if self.is_headless == False and hasattr(self, 'geometry_helper'):
            self.geometry_helper.draw()

    def set_up_scene(self, scene) -> None:
        # super().set_up_scene(scene)
        collision_filter_global_paths = []

        if self.asset.is_add_ground_plane:
            scene.add_default_ground_plane(prim_path='/ground')
            collision_filter_global_paths.append('/ground')

        for name, usd in self.obstacles.items():
            add_usd(usd)
            collision_filter_global_paths.append(usd.prim_path)

        # 添加第0号克隆体
        add_usd(self.pumbaa_usd, self.default_zero_env_path + '/' + self.pumbaa_usd.name)
        self._pumbaa_robot = PumbaaRobot(f"{self.default_zero_env_path}/{self.pumbaa_usd.name}",
                                         name="pumbaa_robot")
        scene.add(self._pumbaa_robot)

        if self.num_envs > 1:

            # 开始克隆
            prim_paths = self._cloner.generate_paths("/World/envs/env", self.num_envs)
            self.env_pos = self._cloner.clone(source_prim_path="/World/envs/env_0", prim_paths=prim_paths,
                                          replicate_physics=True)

            # 为克隆后的机器人添加碰撞过滤
            self._cloner.filter_collisions(
                World.instance().get_physics_context().prim_path, "/World/collisions", prim_paths,
                collision_filter_global_paths)

        # 添加机器人所需要用到的view
        self.robot_view = PumbaaRobotView(f'/World/envs/.*/{self.pumbaa_usd.name}')
        scene.add(self.robot_view)

        self.articulation_view = PumbaaArticulationView(f"/World/envs/.*/{self.pumbaa_usd.name}",
                                                        name="pumbaa_view", reset_xform_properties=False)
        scene.add(self.articulation_view)

        self.base_link_view = PumbaaBaseLinkView(f"/World/envs/.*/{self.pumbaa_usd.name}",
                                                 name='pumbaa_base_xfrom')
        scene.add(self.base_link_view)

        # 添加接触点传感器
        self.baselink_prim_paths = find_matching_prim_paths(f'/World/envs/.*/{self.pumbaa_usd.name}')
        self.baselink_contacts = []
        self.flipper_contacts = []
        for path in self.baselink_prim_paths:
            baselink_contact = PumbaaBaselinkContact(path, collision_filter_global_paths)
            flipper_contact = PumbaaFlipperContact(path, collision_filter_global_paths)
            self.baselink_contacts.append(baselink_contact)
            self.flipper_contacts.append(flipper_contact)

        if self.asset.has_camera():
            self.set_initial_camera_params(**self.asset.camera)

        if self.device != 'cpu':
            for name, usd in self.obstacles.items():
                update_collision(usd, is_to_convex=True)


    def set_initial_camera_params(self, position=[10, 10, 3], target=[0, 0, 0]):
        if self.is_headless == False:
            set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")

    def cleanup(self) -> None:
        super().cleanup()
        self.num_steps = torch.zeros((self._num_envs, ), device=self._device, dtype=torch.int32)

        self.start_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.orientations = torch.zeros((self.num_envs, 4), device=self._device)
        self.orientations_3 = torch.zeros((self.num_envs, 3), device=self._device)

        self.trajectories = [[] for _ in range(self.num_envs)]
        self.angles = [[] for _ in range(self.num_envs)]

        if self._is_record_extras_info == True:
            self.extras['end_type'] = dict()
            self.extras['reward_componets'] = dict()

        self._end_type_list = deque(maxlen=self.num_envs * 2)
        self._total_rewards = torch.zeros((self.num_envs, ), device=self.device)
        self._history_rewards = deque(maxlen=1000)
        self._reward_value_dict = {i.__name__:deque(maxlen=self.num_envs * 1) for i in self.reward_componets}

    def init_task(self):
        self.is_headless = self._cfg['headless']
        self.cleanup()

        config_path = self._task_cfg["asset"]
        self.asset = AssetEntry(file=config_path)
        if hasattr(self, '_reset_info_maps_file'):
            with open(self._reset_info_maps_file, 'r') as maps_stream:
                info_file = yaml.safe_load(maps_stream)[os.path.basename(config_path)]
                # print(info_file)
            with open(info_file, 'rb') as info_stream:
                self._reset_info_dict = pickle.load(info_stream)

        self.pumbaa_usd = self.asset.robot

        self.obstacles = self.asset.obstacles

        self.map_helper = MapHelper(**self.asset.map, rebuild=False)

        if self.asset.has_geometry():
            self.geometry_helper = Geometryhelper(**self.asset.geometry)


        self.spacing = self._env_spacing if self._env_spacing is not None else 5
        self._cloner = GridCloner(spacing=self.spacing)
        self._cloner.define_base_env(self.default_base_env_path)


    def init_gym(self):

        self.flipper_dt = 2

        if not hasattr(self, 'max_v'):
            self.max_v = 0.55

        self.set_action_mode()

        gym_info = self._action_mode_execute.get_gym_info()

        self._num_actions = gym_info['num_actions']
        self.action_space = gym_info['space']

    def init_field(self):
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]


        for field, key in self._field_to_key_maps.items():

            if hasattr(self, field):
                continue

            if key in self._task_cfg['env']:
                value = self._task_cfg['env'][key]
            elif key in self._default_values:
                value = self._default_values[key]
            elif field in self._default_values:
                value = self._default_values[field]
            else:
                logger.warning(f'could not find {key}')
                continue

            setattr(self, field, value)

    @property
    def max_step(self):
        return self._max_episode_length

    @property
    def action_mode(self):
        return ActionMode(self._action_mode)
    def set_action_mode(self, action_mode=None):
        if action_mode is None:
            action_mode = self.action_mode
        else:
            self._action_mode = action_mode

        self._action_mode_execute = ActionModeExecute(action_mode, self.max_v, self.flipper_dt)





