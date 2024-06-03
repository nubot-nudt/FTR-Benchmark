import torch

from omni.isaac.core.utils.rotations import quat_to_euler_angles, euler_angles_to_quat

from .rl_base import AutoConfigRLTask
from .crossing_task import *


class BenchmarkTask(CrossingTask):
    def __init__(
            self,
            name,
            sim_config,
            env,
            offset=None
    ) -> None:
        CrossingTask.__init__(self, name, sim_config, env, offset)

        self.cfg['env'] = {
            'env_name': self.cfg['experiment'].split('/')[-1],
        }

    def get_states(self):
        return self.obs_buf

    def get_extras(self):
        return dict()

    def cleanup(self) -> None:
        super().cleanup()
        self.extras = dict()

    def post_physics_step(self):
        # 计算 obs、reward、done 需要的数据
        self.positions, self.orientations = self.pumbaa_robots.get_world_poses()
        self.flipper_positions = self.pumbaa_robots.get_all_flipper_positions()
        world_vels = self.pumbaa_robots.get_velocities()
        self.velocities[:, :3] = quat_rotate_inverse(self.orientations, world_vels[:, :3])
        self.velocities[:, 3:] = quat_rotate_inverse(self.orientations, world_vels[:, 3:])

        self.orientations_3 = torch.stack(
            list(
                torch.from_numpy(quat_to_euler_angles(i)).to(self.device) for i in self.orientations.cpu()
            )
        )

        # 记录历史信息
        for i in range(self.num_envs):
            self.history_positions[i].append(self.positions[i])
            # self.history_orientations_3[i].append(torch.rad2deg(self.orientations_3[i]))
            # self.history_flippers[i].append(self.flipper_positions[i])
            # self.history_actions[i].append(self.actions[i])
            # self.history_velocities[i].append(torch.clone(self.velocities[i]))

        return AutoConfigRLTask.post_physics_step(self)
