import datetime
import os
import re
import ast
import pickle
import base64
import sys
from contextlib import contextmanager
from pprint import pprint
import traceback
import gym

from ptask_common import project_dir_join, isaac_python_path, project_dir
from omegaconf import OmegaConf
from ptask_envs.omniisaacgymenvs.utils.config_utils.path_utils import retrieve_checkpoint_path
from ptask_envs.omniisaacgymenvs.envs.vec_env_rlgames import VecEnvRLGames

from hydra._internal.hydra import Hydra
from hydra._internal.utils import create_automatic_config_search_path, get_args_parser
from hydra.types import RunMode

from ptask_common.utils.reformat import omegaconf_to_dict
from ptask_common.utils.shell import command_result
from ptask_common.utils.common.asset import check_and_map


class PumbaaVecEnv(VecEnvRLGames):

    def __init__(self, headless: bool, sim_device: int = 0, enable_livestream: bool = False,
                 enable_viewport: bool = False) -> None:
        super().__init__(headless, sim_device, enable_livestream, enable_viewport)

    def get_reward_infos(self):
        return self._task.reward_infos

    @property
    def task(self):
        return self._task

    @property
    def simulation_app(self):
        return self._simulation_app


OmegaConf.register_new_resolver('eq', lambda x, y: x.lower() == y.lower())
OmegaConf.register_new_resolver('contains', lambda x, y: x.lower() in y.lower())
OmegaConf.register_new_resolver('if', lambda pred, a, b: a if pred else b)
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg == '' else arg)
OmegaConf.register_new_resolver('len', lambda x: len(x))


def load_hydra_config(convert_dict=False):
    config_path = project_dir_join('cfg')
    config_file = 'config'
    args = get_args_parser().parse_args()
    search_path = create_automatic_config_search_path(config_file, None, config_path)
    hydra_object = Hydra.create_main_hydra2(task_name='load_omniisaacgymenv', config_search_path=search_path)
    cfg = hydra_object.compose_config(config_file, args.overrides, run_mode=RunMode.RUN)
    del cfg.hydra

    if convert_dict:
        return omegaconf_to_dict(cfg)
    else:
        return cfg


def load_config_by_args(args=''):
    if isinstance(args, list) or isinstance(args, tuple):
        args = ' '.join(args)

    ret = command_result('{} src/load_config.py {}'.format(isaac_python_path, args))
    # print(ret)
    try:
        ret = re.findall(r"b'.*'", ret)[0]
    except IndexError:
        print(ret)
        traceback.print_exc()
        sys.exit(1)

    cfg_b = ast.literal_eval(ret)
    cfg = pickle.loads(base64.b64decode(cfg_b))
    return cfg


def start_only_isaacgym_by_args(args):
    cfg_dict = load_config_by_args(args)
    headless = cfg_dict['headless']
    enable_viewport = "enable_cameras" in cfg_dict['task']['sim'] and cfg_dict['task']['sim']['enable_cameras']
    env = PumbaaVecEnv(
        headless=headless,
        sim_device=cfg_dict['device_id'],
        enable_livestream=cfg_dict['enable_livestream'],
        enable_viewport=enable_viewport
    )

    from ptask_envs.pumbaa import initialize_task
    task = initialize_task(cfg_dict, env)

    from ptask_envs.envs.register import wrap_envs
    env = wrap_envs(cfg_dict, env)

    return env, task

@contextmanager
def launch_isaacgym_env(preprocess_func=None, check_map=True):
    # from hydra.core.global_hydra import GlobalHydra
    #
    # hydra = GlobalHydra.instance()
    # if hydra.is_initialized():
    #     hydra.clear()

    cfg = load_hydra_config()

    cfg_dict = omegaconf_to_dict(cfg)

    if preprocess_func is not None:
        preprocess_func(config=cfg_dict)
    pprint(cfg_dict)

    if check_map:
        check_and_map(cfg_dict['task']['asset'])

    headless = cfg_dict['headless']

    time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    rank = int(os.getenv("LOCAL_RANK", "0"))
    if cfg.multi_gpu:
        cfg.device_id = rank
        cfg.rl_device = f'cuda:{rank}'

    enable_viewport = "enable_cameras" in cfg.task.sim and cfg.task.sim.enable_cameras
    env = PumbaaVecEnv(
        headless=headless,
        sim_device=cfg.device_id,
        enable_livestream=cfg.enable_livestream,
        enable_viewport=enable_viewport or cfg.enable_recording,
    )

    # 启动扩展
    from omni.isaac.core.utils.extensions import enable_extension
    enable_extension('omni.isaac.sensor')

    experiment_name = cfg.experiment or cfg.train.params.config.name or cfg.train.params.config.full_experiment_name
    if not experiment_name.startswith('runs'):
        experiment_name = os.path.join('runs', experiment_name)
    experiment_dir = os.path.join(project_dir, experiment_name)

    # use gym RecordVideo wrapper for viewport recording
    if cfg.enable_recording:
        if cfg.recording_dir == '':
            videos_dir = os.path.join(experiment_dir, "videos")
        else:
            videos_dir = cfg.recording_dir
        video_interval = lambda step: step % cfg.recording_interval == 0
        video_length = cfg.recording_length
        env.is_vector_env = True
        if env.metadata is None:
            env.metadata = {"render_modes": ["rgb_array"], "render_fps": cfg.recording_fps}
        else:
            env.metadata["render_modes"] = ["rgb_array"]
            env.metadata["render_fps"] = cfg.recording_fps
        env = gym.wrappers.RecordVideo(
            env, video_folder=videos_dir, step_trigger=video_interval, video_length=video_length
        )

    if cfg.checkpoint:
        cfg.checkpoint = retrieve_checkpoint_path(cfg.checkpoint)
        if cfg.checkpoint is None:
            quit()

    from omni.isaac.core.utils.torch.maths import set_seed
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic)
    cfg_dict['seed'] = cfg.seed

    from ptask_envs.pumbaa import initialize_task
    task = initialize_task(cfg_dict, env)

    from ptask_envs.envs.register import wrap_envs
    env = wrap_envs(cfg_dict, env)

    if cfg.wandb_activate and rank == 0:
        # Make sure to install WandB if you actually use this.
        import wandb

        run_name = f"{cfg.wandb_name}_{time_str}"

        wandb.init(
            project=cfg.wandb_project,
            group=cfg.wandb_group,
            entity=cfg.wandb_entity,
            config=cfg_dict,
            sync_tensorboard=True,
            name=run_name,
            resume="allow",
        )

    yield {
        'omegaconf': cfg,
        'config': cfg_dict,
        'env': env,
        'task': task
    }

    # env.close()
    if cfg.wandb_activate and rank == 0:
        wandb.finish()
