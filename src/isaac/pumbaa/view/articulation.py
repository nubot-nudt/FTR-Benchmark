from typing import Optional

import torch

from omni.isaac.core.articulations import ArticulationView

from pumbaa.common.default import *

class PumbaaArticulationView(ArticulationView):


    def __init__(
        self,
        base_prim_paths_expr: str,
        name: Optional[str] = "PumbaaArticulationView",
        reset_xform_properties = False
    ) -> None:
        '''
        该类为控制救援机器人默认ArticulationView，包括摆臂控制和差速控制
        :param prim_paths_expr:
        :param name:
        :param reset_xform_properties:
        '''

        super().__init__(
            prim_paths_expr=f"{base_prim_paths_expr}/{robot_prim_path}",
            name=name,
            reset_xform_properties=reset_xform_properties
        )

    def get_all_flipper_efforts(self, indices=None):
        return self.get_applied_joint_efforts(indices=indices, joint_indices=self.flipper_dof_idx_list)

    def get_all_flipper_joint_velocities(self, indices=None):
        return self.get_joint_velocities(indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_all_flipper_dt(self, dt_positions: torch.Tensor, indices=None, max_position=60):

        current_pos = self.get_all_flipper_positions(indices=indices)
        a = torch.tensor([-1, -1, 1, 1], device=dt_positions.device)

        pos = current_pos + dt_positions

        pos = torch.clip(pos, -max_position, max_position)

        pos = pos * a * torch.pi / 180

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def get_all_flipper_positions(self, indices=None):
        positions = self.get_joint_positions(indices=indices, joint_indices=self.flipper_dof_idx_list)

        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        return torch.rad2deg(positions * a)

    def set_all_flipper_positions_with_dt(self, positions: torch.Tensor, dt, indices=None):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = positions

        current_pos = self.get_all_flipper_positions(indices=indices)
        for i in range(len(current_pos)):
            for j in range(4):
                if current_pos[i][j] > pos[i][j]:
                    pos[i][j] = torch.max(pos[i][j], current_pos[i][j] - dt)
                else:
                    pos[i][j] = torch.min(pos[i][j], current_pos[i][j] + dt)

        pos = pos * a * torch.pi / 180

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)


    def set_all_flipper_position_targets(self, positions: torch.Tensor, indices=None):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = torch.deg2rad(positions * a)

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_all_flipper_positions(self, positions: torch.Tensor, indices=None):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = torch.deg2rad(positions * a)

        self.set_joint_positions(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)


    def set_v_w(self, v_w: torch.Tensor, indices=None):
        v = v_w[:, 0]
        w = v_w[:, 1]

        vels = torch.zeros(v_w.shape, device=v_w.device)
        vels[:, 0] = (2 * v + w * L) / 2
        vels[:, 1] = (2 * v - w * L) / 2

        # print('vels', vels)

        return self.set_right_and_left_velocities(vels, indices=indices)

    def get_v_w(self, indices=None):
        v_r, v_l = self.get_right_and_left_velocities(indices=indices)

        return (v_r + v_l) / 2, (v_r - v_l) / L

    def set_right_and_left_velocities(self, vels, indices=None):
        r_v = torch.ones((vels.shape[0], len(self.r_indices)), device=vels.device)
        l_v = torch.ones((vels.shape[0], len(self.l_indices)), device=vels.device)
        fr_v = torch.ones((vels.shape[0], len(self.fr_indices)), device=vels.device)
        fl_v = torch.ones((vels.shape[0], len(self.fl_indices)), device=vels.device)

        for i in range(vels.shape[0]):
            r_v[i, :] *= -vels[i, 0] / wheel_radius
            l_v[i, :] *= -vels[i, 1] / wheel_radius
            fr_v[i, :] *= vels[i, 0] / wheel_radius
            fl_v[i, :] *= vels[i, 1] / wheel_radius

        self.set_joint_velocities(r_v, indices=indices, joint_indices=self.r_indices)
        self.set_joint_velocities(l_v, indices=indices, joint_indices=self.l_indices)
        self.set_joint_velocities(fr_v, indices=indices, joint_indices=self.fr_indices)
        self.set_joint_velocities(fl_v, indices=indices, joint_indices=self.fl_indices)

    def get_right_and_left_velocities(self, indices=None, ):
        r_vels = self.get_joint_velocities(indices=indices, joint_indices=self.r_indices)
        l_vels = self.get_joint_velocities(indices=indices, joint_indices=self.l_indices)

        v_r = r_vels.sum(axis=1) / r_vels.shape[1] * wheel_radius
        v_l = l_vels.sum(axis=1) / l_vels.shape[1] * wheel_radius

        return v_r, v_l

    def find_idx(self):
        self.flipper_dof_idx_list = [self.get_dof_index(i) for i in flipper_joint_names]
        self.wheel_dof_idx_list = [self.get_dof_index(i) for i in wheel_joint_names]

        self.r_indices = [self.get_dof_index(i) for i in baselink_wheel_joint_names if i.startswith('R')]
        self.l_indices = [self.get_dof_index(i) for i in baselink_wheel_joint_names if i.startswith('L')]

        self.fr_indices = [self.get_dof_index(i) for i in flipper_wheel_joint_names if i.startswith('R')]
        self.fl_indices = [self.get_dof_index(i) for i in flipper_wheel_joint_names if i.startswith('L')]


