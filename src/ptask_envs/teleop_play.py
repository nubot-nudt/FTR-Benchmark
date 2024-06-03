import fire
from omni.isaac.gym.vec_env import VecEnvBase
from ptask_common.utils.tui.select_info import select_asset_path


class PumbaaVecEnv(VecEnvBase):

    def get_reward_infos(self):
        return self._task.reward_infos

    @property
    def task(self):
        return self._task

    @property
    def simulation_app(self):
        return self._simulation_app


def main(config_path: str = None):
    if config_path is None:
        config_path = select_asset_path()
    env = PumbaaVecEnv(headless=False)

    from omni.isaac.core.utils.extensions import enable_extension
    from ptask_envs.tasks.base import PumbaaBaseTask
    from ptask_envs.pumbaa.utils.asset import AssetEntry
    from ptask_envs.envs.interface.keyboard import KeyboardEnv

    enable_extension('omni.isaac.sensor')
    task = PumbaaBaseTask(name="Pumbaa", asset=AssetEntry(file=config_path))
    env.set_task(task, backend="torch")
    env._world.reset()
    env = KeyboardEnv(env, config_path)
    env.run()
    env.close()


if __name__ == '__main__':
    fire.Fire(main)












