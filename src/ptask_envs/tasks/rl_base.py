
import inspect
from abc import abstractmethod
from collections import deque
from itertools import product


import torch
from loguru import logger
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from ptask_envs.omniisaacgymenvs.tasks.base.rl_task import RLTask

from ptask_envs.pumbaa.utils.asset import AssetEntry
from ptask_envs.pumbaa.utils.map import MapHelper
from ptask_envs.pumbaa.utils.prim import add_usd
from ptask_common.utils.tensor import to_numpy, to_tensor
from ptask_common.processing.perception.height_map import BigMapConvertor


def orient_4to3(orient):
    return to_tensor(quat_to_euler_angles(to_numpy(orient)))


class AutoConfigRLTask(RLTask):

    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        self.init_config(sim_config)

        self.is_play = self._sim_config.config['test']
        self.is_debug = self._sim_config.config['debug']
        self.done_num = 0
        logger.info(f'start with play={self.is_play} debug={self.is_debug}')

        self.init_field()
        self.init_gym()
        RLTask.__init__(self, name, env)
        self.init_task()

    def init_config(self, sim_config):
        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config
        if not hasattr(self, 'task_config'):
            self.task_config = self._task_cfg

    def get_observations(self):

        index = 0
        for name, d in self.observation_components.items():
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

    def post_reset_idx(self, env_ids):
        if self.is_play:
            self.trajectory_rewards[env_ids] = 0

    def post_physics_step(self):
        obs_buf, rew_buf, reset_buf, extras = super().post_physics_step()
        reset_buf = torch.sign(reset_buf)
        self.physics_step_num += 1

        return obs_buf, rew_buf, reset_buf, extras

    def prepare_to_calculate_metrics(self):
        if not hasattr(self, '_reward_func_deps_list'):
            self._reward_func_deps_list = []

            for reward_name, obs_name in self.reward_func_deps.items():
                if reward_name not in self.reward_components:
                    continue
                if obs_name in self.observation_components:
                    continue

                self._reward_func_deps_list.append(self.observation_components[obs_name]['func'])

        for func in self._reward_func_deps_list:
            func()

    def _compute_reward(self, env_i, rew_i, rew_name, rew_coef, rew_func=None):

        if rew_func is None:
            rew_func = self.reward_components[rew_name]['func']

        rew_value = rew_func(env_i)
        rew_weight_value = rew_value * rew_coef

        return rew_value, rew_weight_value

    # @log_mean_run_time(mean_num=10)
    def calculate_metrics(self) -> None:
        self.rew_buf[:] = 0
        self.reward_vec_buf[:, :] = 0
        self.prepare_to_calculate_metrics()

        if not hasattr(self, '_compute_reward_params'):
            # (env_i, rew_i, rew_name, rew_coef, rew_func, args_num)
            self._compute_reward_params = [
                (env_i, r_comp[0], r_comp[1][0], r_comp[1][1]['coef'], r_comp[1][1]['func'], r_comp[1][1]['args_num'])
                for env_i, r_comp in product(range(self.num_envs), enumerate(self.reward_components.items()))
            ]
            self._loop_compute_reward_params = list(filter(lambda x:x[-1] > 0, self._compute_reward_params))
            self._vec_compute_reward_params = [
                (r_comp[0], r_comp[1][0], r_comp[1][1]['coef'], r_comp[1][1]['func'])
                for r_comp in enumerate(self.reward_components.items())
                if r_comp[1][1]['args_num'] == 0
            ]

        for rew_i, rew_name, rew_coef, rew_func in self._vec_compute_reward_params:
            self.reward_vec_buf[:, rew_i] = rew_func()

        for env_i, rew_i, rew_name, rew_coef, rew_func, args_num in self._loop_compute_reward_params:

            try:
                rew_value, rew_weight_value = self._compute_reward(env_i, rew_i, rew_name, rew_coef, rew_func)
            except Exception as e:
                logger.error(f'{rew_name} : {type(e)} {e}')
                raise e

            self.reward_vec_buf[env_i, rew_i] = rew_value

            if torch.abs(self.reward_vec_buf[env_i, rew_i]) > 1e5:
                logger.warning(f'{rew_name}={rew_weight_value} is abnormal value in calculate_metrics')

        self.reward_infos = []
        if self._task_cfg['task']['record_reward_item']:
            for env_i, rew_i, rew_name, rew_coef, _, _ in self._compute_reward_params:
                self._reward_value_dict[rew_name].append(self.reward_vec_buf[env_i, rew_i])
                self.reward_infos.append({
                    'index': env_i,
                    'name': rew_name,
                    'reward': self.reward_vec_buf[env_i, rew_i].item(),
                    'coef': rew_coef,
                })

        if not hasattr(self, '_reward_coef_tensor'):
            self._reward_coef_tensor = torch.tensor([i['coef'] for i in self.reward_components.values()])

        self.rew_buf[:] = torch.sum(self.reward_vec_buf * self._reward_coef_tensor, dim=-1)
        if self.is_play:
            self.trajectory_rewards += self.rew_buf

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


    @abstractmethod
    def take_actions(self, actions, indices):
        pass

    @abstractmethod
    def reset_idx(self, env_ids):
        pass

    def reset(self):
        super().reset()
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)

        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

    def get_extras(self):
        if len(self._end_type_list) > 0:
            for end_type in [t for t in self.done_components]:
                self.extras['end_type']['end_' + end_type] = self._end_type_list.count(end_type) / len(
                    self._end_type_list)

        for name, value in self._reward_value_dict.items():
            self.extras['reward_components'][name] = 0 if len(value) == 0 else sum(value) / len(value)

        return super().get_extras()

    def set_up_scene(self, scene, collision_filter_global_paths=[]) -> None:
        # super().set_up_scene(scene)
        self._collision_filter_global_paths = []

        if self.asset.is_add_ground_plane:
            scene.add_default_ground_plane(prim_path='/ground')
            self._collision_filter_global_paths.append('/ground')

        # add terrain and obstacle to world
        for name, usd in self.obstacles.items():
            add_usd(usd)
            self._collision_filter_global_paths.append(usd.prim_path)
        for name, terrain in self.asset.terrains.items():
            p = terrain.add_terrain_to_stage(scene.stage)
            self._collision_filter_global_paths.append(p)
        super().set_up_scene(scene, collision_filter_global_paths + self._collision_filter_global_paths)

    def cleanup(self) -> None:
        super().cleanup()
        self.physics_step_num = 0
        self.num_steps = torch.zeros((self._num_envs,), device=self._device, dtype=torch.int32)

        if self.is_play:
            self.trajectory_rewards = torch.zeros((self.num_envs,), device=self.device)

        self.reward_vec_buf = torch.zeros((self._num_envs, self.reward_dim), device=self._device, dtype=torch.float)

        self.extras['end_type'] = dict()
        self.extras['reward_components'] = dict()

        self._end_type_list = deque(maxlen=self.num_envs * 2)
        self._reward_value_dict = {i: deque(maxlen=self.num_envs * 1) for i in self.reward_components}

    @abstractmethod
    def prepare_reset_info(self):
        pass

    def init_task(self):
        self.cleanup()
        self.asset = AssetEntry(file=self._task_cfg["asset"])
        self.hmap_helper = MapHelper(**self.asset.map, rebuild=False)
        self.hmap_convertor = BigMapConvertor(
            height_map_array=self.hmap_helper.map,
            lower=self.asset.map['lower'],
            upper=self.asset.map['upper'],
            cell_size=self.asset.map['cell_size'],
        )
        self.prepare_reset_info()
        self.obstacles = self.asset.obstacles
        self.spacing = self._env_spacing if self._env_spacing is not None else 5
        # self._cloner = GridCloner(spacing=self.spacing)
        # self._cloner.define_base_env(self.default_base_env_path)

    def register_gym_func(self):
        if not hasattr(self, 'reward_func_deps'):
            self.reward_func_deps = dict()
        if not hasattr(self, 'observation_component_register'):
            self.observation_component_register = dict()
        if not hasattr(self, 'reward_component_register'):
            self.reward_component_register = dict()
        if not hasattr(self, 'done_components'):
            self.done_components = dict()

    def init_gym(self):
        self.register_gym_func()
        
        # state
        self.observation_components = self._load_state_components(self.task_config.state_space_dict)
        self._num_observations = sum([i['size'] for i in self.observation_components.values()])
        
        if self.task_config.hidden_state_space_dict is None or len(self.task_config.hidden_state_space_dict) == 0:
            self._num_states = self._num_observations
        else:
            self.hidden_observation_components = self._load_state_components(self.task_config.hidden_state_space_dict)
            self._num_states = sum([i['size'] for i in self.hidden_observation_components.values()])

        # reward
        self.reward_components = dict()
        for name, coef in self._reward_coef.items():
            if hasattr(self, f'_calculate_metrics_{name}'):
                func = getattr(self, f'_calculate_metrics_{name}')
            elif hasattr(self, f'calculate_metrics_{name}'):
                func = getattr(self, f'calculate_metrics_{name}')
            elif hasattr(self, 'reward_component_register'):
                func = self.reward_component_register[name]
            else:
                raise KeyError(f'{name} does not exist')

            self.reward_components[name] = {
                'coef': coef,
                'func': func,
                'args_num': len(inspect.signature(func).parameters),
            }

    def _load_state_components(self, state_space_dict):
        components = dict()
        for name, item in state_space_dict.items():
            if hasattr(self, f'_get_observations_{name}'):
                func = getattr(self, f'_get_observations_{name}')
            elif hasattr(self, f'get_observations_{name}'):
                func = getattr(self, f'get_observations_{name}')
            elif hasattr(self, name):
                func = (lambda: getattr(self, name))
            elif hasattr(self, 'observation_component_register'):
                func = self.observation_component_register[name]
            else:
                raise KeyError(f'{name} does not exist')

            if isinstance(item, int):
                components[name] = {
                    'size': item,
                    'func': func,
                }
            elif isinstance(item, dict):
                components[name] = {
                    'func': func,
                    **item,
                }
            else:
                raise RuntimeError(f'{name} is not support')

        return components

    _field_to_keys = []

    def init_field(self):

        self.is_headless = self._cfg['headless']
        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]

        # 更新字段，将task_config的字段设置为自己的保护成员
        # 根据 _field_to_keys 查找，查找顺序【 保护成员， task_config 】
        for field in self._field_to_keys:

            if hasattr(self, f'_{field}') or hasattr(self, field):
                continue

            if hasattr(self.task_config, field) and getattr(self.task_config, field) is not None:
                setattr(self, f'_{field}', getattr(self.task_config, field))
                continue

            raise KeyError(f'{field} does not exist')

    def is_done(self) -> None:
        for i in range(self.num_envs):
            for t, f in self.done_components.items():
                if f(i):
                    self.reset_buf[i] += 1

                    if self._task_cfg['task']['record_done']:
                        self._end_type_list.append(t)

                    if self.is_play or self.is_debug:
                        print(f'** {i}-th end with {t}, reward is {self.trajectory_rewards[i]}')

                    self.done_num += 1
                    if self.done_num == self.num_envs:
                        logger.info(f'{self.done_num} round has been completed')

        self.reset_buf = torch.clip(self.reset_buf, 0, 1)

    @property
    def max_step(self):
        return self._max_episode_length

    def set_camera_follow(self, position):
        cp = position + torch.tensor([0, 4, 1.5])
        tp = position + torch.tensor([0, -2, -0.5])
        self.set_initial_camera_params(
            cp.tolist(),
            tp.tolist(),
        )

    @property
    def current_time(self):
        return self.world.current_time

    @property
    def reward_dim(self):
        return len(self.reward_components)
