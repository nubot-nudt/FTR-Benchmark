from itertools import repeat, chain
from typing import List


import torch
from omni.isaac.core.prims import RigidPrimView, RigidContactView

from pumbaa.common.default import *


class PumbaaRigidPrimView(RigidPrimView):

    def __init__(
            self,
            prim_paths_expr: str,
            name: str = "pumbaa_rigid_prim_view"
    ):
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            name=name,

        )


class PumbaaRigidContactView(RigidContactView):

    def __init__(
            self,
            prim_paths_expr: str,
            filter_paths_expr: List[str],
            name: str = "pumbaa_rigid_contact_view"
    ):
        self.name = name
        super().__init__(
            prim_paths_expr=prim_paths_expr,
            filter_paths_expr=filter_paths_expr,
            name=name,
            prepare_contact_sensors=True,
            apply_rigid_body_api=False,
        )

class PumbaaBaselinkContact(PumbaaRigidContactView):
    def __init__(
            self,
            base_prim_path: str,
            filter_paths_expr: List[str],
            name: str = "pumbaa_baselink_contact"
    ):
        # print(find_matching_prim_paths(f'{base_prim_path}/{baselink_wheel_prim_path}'))
        super().__init__(
            prim_paths_expr=f'{base_prim_path}/{baselink_wheel_prim_path}',
            filter_paths_expr=filter_paths_expr,
            name=name
        )

    baselink_index_to_point_maps = torch.tensor([
        # L
        *[[i, 0.24] for i in [0.28, 0.2, 0.12, 0.04, -0.04, -0.12, -0.2 , -0.28]],
        # R
        *[[i, -0.24] for i in [0.28, 0.2, 0.12, 0.04, -0.04, -0.12, -0.2 , -0.28]],
    ])
    def get_contact_points(self):
        forces = self.get_net_contact_forces()

        indexes = (forces[:, 2] > 0).nonzero().squeeze(-1)

        return self.baselink_index_to_point_maps[indexes.cpu()].to(indexes.device)

    def get_left_and_right_forces(self):
        forces = self.get_net_contact_forces()
        return forces[:8], forces[9:]

class PumbaaFlipperContact(PumbaaRigidContactView):
    def __init__(
            self,
            base_prim_path: str,
            filter_paths_expr: List[str],
            name: str = "pumbaa_baselink_contact"
    ):
        # print(find_matching_prim_paths(f'{base_prim_path}/{flipper_prim_path}'))
        super().__init__(
            prim_paths_expr=f'{base_prim_path}/{flipper_prim_path}',
            filter_paths_expr=filter_paths_expr,
            name=name
        )

    flipper_index_to_point_maps = torch.tensor([
        # FL
        *[[0.28 + i, 0.28] for i in [0, 0.085, 0.17, 0.255, 0.34]],
        # FR
        *[[0.28 + i, -0.28] for i in [0, 0.085, 0.17, 0.255, 0.34]],
        # RL
        *[[-0.62 + i, 0.28] for i in [0, 0.085, 0.17, 0.255, 0.34]],
        # RR
        *[[-0.62 + i, -0.28] for i in [0, 0.085, 0.17, 0.255, 0.34]],
    ])
    def get_contact_points(self):
        forces = self.get_net_contact_forces()

        indexes = (forces[:, 2] > 0).nonzero().squeeze(-1)

        return self.flipper_index_to_point_maps[indexes.cpu()].to(indexes.device)

    def get_left_and_right_forces(self):
        forces = self.get_net_contact_forces()

        indices = torch.tensor(list(chain(repeat(0, 5), repeat(1, 5))) * 2)
        return forces[indices == 0], forces[indices == 1]