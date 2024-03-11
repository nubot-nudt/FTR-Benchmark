
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()


if len(sys.argv) == 1:
    from tui.select_info import select_asset_path
    config_path = select_asset_path()

elif len(sys.argv) == 2:
    config_path = sys.argv[1]

else:
    sys.exit(1)


from omni.isaac.gym.vec_env import VecEnvBase

class PumbaaVecEnv(VecEnvBase):

    def get_reward_infos(self):
        return self._task.reward_infos
    @property
    def task(self):
        return self._task

    @property
    def simulation_app(self):
        return self._simulation_app


env = PumbaaVecEnv(headless=False)

from omni.isaac.core.utils.extensions import enable_extension

enable_extension('omni.isaac.sensor')

from tasks.base import PumbaaBaseTask
from pumbaa.common import AssetEntry


task = PumbaaBaseTask(name="Pumbaa", asset=AssetEntry(file=config_path))
env.set_task(task, backend="torch")

env._world.reset()

from isaacgym_ext.interface.keyboard import KeyboardEnv
env = KeyboardEnv(env)

env.run()

env.close()
