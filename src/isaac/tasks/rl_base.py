import os
import pickle
import random
from abc import abstractmethod
from itertools import cycle


import numpy as np
import yaml
from collections import deque

from loguru import logger

from omni.isaac.cloner import GridCloner
from omni.isaac.core.utils.prims import find_matching_prim_paths
from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.world import World

from omniisaacgymenvs.tasks.base.rl_task import RLTask

from pumbaa.common.asset import AssetEntry
from pumbaa.common.config import PumbaaTaskConfig
from processing.action_mode import ActionModeFactory
from pumbaa.helper.map import MapHelper
from processing.control.flipper import FlipperControl
from isaacsim_ext.prim import add_usd, update_collision
from pumbaa.view.articulation import PumbaaArticulationView
from pumbaa.view.robot import PumbaaRobot, PumbaaBaseLinkView, PumbaaRobotView
from pumbaa.view.rigid import *
from utils.tensor import to_numpy, to_tensor
from utils.common.birth import load_birth_info


class PumbaaBaseRLTask(RLTask):

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

        self.task_config = PumbaaTaskConfig(self._task_cfg)

        # print(self._sim_config.config)
        self.is_play = (self._sim_config.config['test'] == True)
        self.is_debug = (self._sim_config.config['debug'] == True)
        logger.info(f'start with play={self.is_play} debug={self.is_debug}')

        self.init_field()

        self.init_gym()

        RLTask.__init__(self, name, env)

        self.init_task()

    def prepare_reset_info(self):
        '''
        用于家在机器人初始和结束形象，结果会放入self._reset_info中
        :return:
        '''
        if self.asset.has_task_info():
            self._reset_info = [self.asset.task_info]
        elif hasattr(self, '_reset_info_maps_file'):
            self._reset_info = load_birth_info(os.path.basename(self._task_cfg["asset"]), self._reset_info_maps_file)
        else:
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
        self._reset_info_generate = reset_choice_func[self.task_config.reset_type]

    def _reset_idx_robot_info(self, num_resets):
        pos = torch.zeros((num_resets, 3), device=self._device)
        orient = torch.zeros((num_resets, 4), device=self._device)
        target = torch.zeros((num_resets, 3), device=self._device)

        for i in range(num_resets):
            _t = self._reset_info_generate()
            pos[i] = _t['start_point']
            orient[i] = _t['start_orient']
            target[i] = _t['target_point']

        return {
            'pos': pos,
            'orient': orient,
            'target': target,
        }


    def post_physics_step(self):
        # 计算 obs、reward、done 需要的数据
        self.positions, self.orientations = self.robot_view.get_world_poses()
        self.flipper_positions = self.articulation_view.get_all_flipper_positions()
        self.velocities = self.robot_view.get_velocities()

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
            self.history_velocities[i].append(self.velocities)

        obs_buf, rew_buf, reset_buf, extras = super().post_physics_step()

        reset_buf = torch.sign(reset_buf)

        self.physics_step_num += 1

        return obs_buf, rew_buf, reset_buf, extras


    def reset_idx(self, env_ids):
        num_resets = len(env_ids)
        indices = env_ids.to(dtype=torch.int32)

        if self.is_debug:
            logger.debug(f'reset indices={indices}')

        reset_info = self._reset_idx_robot_info(num_resets)

        self.robot_view.set_velocities(torch.zeros((num_resets, 6), device=self._device), indices=indices)
        self.robot_view.set_world_poses(reset_info['pos'], reset_info['orient'], indices=indices)

        self._flipper_control.zero(index=env_ids)
        self.articulation_view.set_all_flipper_positions(torch.zeros((num_resets, 4), device=self._device), indices=indices)
        self.articulation_view.set_v_w(torch.zeros((num_resets, 2), device=self._device), indices=indices)

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
        self.target_positions[env_ids] = reset_info['target']

        self.post_reset_idx(env_ids)

    def post_reset_idx(self, env_ids):
        ...


    def pre_physics_step(self, actions) -> None:
        if actions is None:
            return

        if not self.world.is_playing():
            return

        self.num_steps += 1

        self.actions = actions

        # 获取需要重置和不需要重置的机器人下标
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        indices = torch.arange(self.num_envs, dtype=torch.int64, device=self._device)

        vels, flippers = self.take_actions(actions, indices)
        self.velocity_commands = vels
        self.flipper_commands = flippers

        if not self.is_headless and self.task_config.is_follow_camera:
            self.set_camera_follow(self.positions[0])

    @abstractmethod
    def take_actions(self, actions, indices):
        pass

    def reset(self):
        super().reset()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def post_reset(self):
        if not hasattr(self, '_init_finished_flag'):
            self.articulation_view.find_idx()

            for contact in chain(self.baselink_contacts, self.flipper_contacts):
                contact.initialize()

            if hasattr(self, 'imus'):
                for imu in self.imus:
                    imu.initialize()

            if self.device == 'cpu':
                for name, usd in self.obstacles.items():
                    update_collision(usd)

            self._init_finished_flag = True

    def set_up_scene(self, scene, apply_imu=False) -> None:
        # super().set_up_scene(scene)
        collision_filter_global_paths = []

        if self.asset.is_add_ground_plane:
            scene.add_default_ground_plane(prim_path='/ground')
            collision_filter_global_paths.append('/ground')

        # add terrain and obstacle to world
        for name, usd in self.obstacles.items():
            add_usd(usd)
            collision_filter_global_paths.append(usd.prim_path)
        for name, terrain in self.asset.terrains.items():
            p = terrain.add_terrain_to_stage(scene.stage)
            collision_filter_global_paths.append(p)

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
        if apply_imu:
            from pumbaa.sensor.imu import IMU
            self.imus = [IMU(path, scene) for path in self.baselink_prim_paths]


        self.baselink_contacts = []
        self.flipper_contacts = []
        for path in self.baselink_prim_paths:
            baselink_contact = PumbaaBaselinkContact(path, collision_filter_global_paths)
            flipper_contact = PumbaaAllFipperContact(path, collision_filter_global_paths)
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
        self.physics_step_num = 0
        self.num_steps = torch.zeros((self._num_envs, ), device=self._device, dtype=torch.int32)

        self.start_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.target_positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.positions = torch.zeros((self.num_envs, 3), device=self._device)
        self.flipper_positions = torch.zeros((self.num_envs, 4), device=self._device)
        self.orientations = torch.zeros((self.num_envs, 4), device=self._device)
        self.orientations_3 = torch.zeros((self.num_envs, 3), device=self._device)
        self.velocities = torch.zeros((self.num_envs, 6), device=self._device)
        self.velocity_commands = torch.zeros((self.num_envs, 2), device=self._device)
        self.flipper_commands = torch.zeros((self.num_envs, 4), device=self._device)

        N = 5
        self.history_positions = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_orientations_3 = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_flippers = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_actions = [deque(maxlen=N) for _ in range(self.num_envs)]
        self.history_velocities = [deque(maxlen=N) for _ in range(self.num_envs)]


    def init_task(self):
        self.is_headless = self._cfg['headless']
        self.cleanup()

        config_path = self._task_cfg["asset"]
        self.asset = AssetEntry(file=config_path)

        self.prepare_reset_info()

        self.pumbaa_usd = self.asset.robot

        self.obstacles = self.asset.obstacles

        self.map_helper = MapHelper(**self.asset.map, rebuild=False)

        self.spacing = self._env_spacing if self._env_spacing is not None else 5
        self._cloner = GridCloner(spacing=self.spacing)
        self._cloner.define_base_env(self.default_base_env_path)

        self._flipper_control = FlipperControl(self.num_envs, device=self._device)

    def register_gym_func(self):
        if not hasattr(self, 'reward_func_deps'):
            self.reward_func_deps = dict()

        if not hasattr(self, 'observation_componet_register'):
            self.observation_componet_register = dict()

        if not hasattr(self, 'reward_componet_register'):
            self.reward_componet_register = dict()

        if not hasattr(self, 'self.done_componets'):
            self.done_componets = dict()

    def init_gym(self):
        self.register_gym_func()
        # state
        self.observation_componets = self._load_state_componets(self.task_config.state_space_dict)

        self._num_observations = sum([i['size'] for i in self.observation_componets.values()])
        if self.task_config.hidden_state_space_dict is None or len(self.task_config.hidden_state_space_dict) == 0:
           self._num_states = self._num_observations
        else:
            self.hidden_observation_componets = self._load_state_componets(self.task_config.hidden_state_space_dict)
            self._num_states = sum([i['size'] for i in self.hidden_observation_componets.values()])

        # action
        if not hasattr(self, 'max_v'):
            self.max_v = self.task_config.max_v
        if not hasattr(self, 'max_w'):
            self.max_w = self.task_config.max_w
        if not hasattr(self, 'flipper_dt'):
            self.flipper_dt = self.task_config.flipper_dt

        self.set_action_mode()

        gym_info = self._action_mode_execute.gym_info

        if not hasattr(self, '_num_actions'):
            self._num_actions = gym_info['num_actions']
            self.action_space = gym_info['space']

        # reward
        self.reward_componets = dict()
        for name, coef in self._reward_coef.items():
            if hasattr(self, f'_calculate_metrics_{name}'):
                func = getattr(self, f'_calculate_metrics_{name}')
            elif hasattr(self, f'calculate_metrics_{name}'):
                func = getattr(self, f'calculate_metrics_{name}')
            elif hasattr(self, 'reward_componet_register'):
                func = self.reward_componet_register[name]
            else:
                raise KeyError(f'{name} does not exist')

            self.reward_componets[name] = {
                'coef': coef,
                'func': func,
            }

    def _load_state_componets(self, state_space_dict):
        componets = dict()
        for name, item in state_space_dict.items():
            if hasattr(self, f'_get_observations_{name}'):
                func = getattr(self, f'_get_observations_{name}')
            elif hasattr(self, f'get_observations_{name}'):
                func = getattr(self, f'get_observations_{name}')
            elif hasattr(self, name):
                func = (lambda: getattr(self, name))
            elif hasattr(self, 'observation_componet_register'):
                func = self.observation_componet_register[name]
            else:
                raise KeyError(f'{name} does not exist')

            if isinstance(item, int):
                componets[name] = {
                    'size': item,
                    'func': func,
                }
            elif isinstance(item, dict):
                componets[name] = {
                    'func': func,
                    **item,
                }
            else:
                raise RuntimeError(f'{name} is not support')

        return componets

    def init_field(self):
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        _field_to_keys = {
            'max_episode_length',
            'state_space_dict',
            'action_mode',
            'reward_coef',
            'reset_info_maps_file',
        }

        for field in _field_to_keys:

            if hasattr(self, f'_{field}') or hasattr(self, field):
                continue

            if hasattr(self.task_config, field) and getattr(self.task_config, field) is not None:
                setattr(self, f'_{field}', getattr(self.task_config, field))
                continue

            raise KeyError(f'{field} does not exist')

    @property
    def max_step(self):
        return self._max_episode_length

    @property
    def action_mode(self):
        return self._action_mode

    def set_action_mode(self, action_mode=None):
        if action_mode is None:
            action_mode = self.action_mode
        else:
            self._action_mode = action_mode

        self._action_mode_execute = ActionModeFactory.get_action_mode_by_name(action_mode, max_v=self.max_v, flipper_dt=self.flipper_dt)

    def set_camera_follow(self, position):
        self.set_initial_camera_params(
            [position[0] + 0, position[1] + 4, position[2] + 2],
            [position[0], position[1] - 1, position[2]]
        )

    def set_initial_camera_params(self, position=[10, 10, 3], target=[0, 0, 0]):
        set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")

    @property
    def current_time(self):
        return self.world.current_time