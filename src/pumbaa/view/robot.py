

from typing import Optional
import numpy as np
from omni.isaac.core.prims import XFormPrimView
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.robots import RobotView

from pumbaa.common.default import *

class PumbaaRobot(Robot):
    def __init__(
        self,
        base_prim_path: str,
        name: Optional[str] = "PumbaaRobot",
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._name = name
        super().__init__(
            prim_path=f'{base_prim_path}/{robot_prim_path}',
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )


class PumbaaRobotView(RobotView):
    def __init__(
            self,
            base_prim_paths_expr: str,
            name: str = "PumbaaRobotView"
    ):
        self._name = name
        super().__init__(
            prim_paths_expr=f'{base_prim_paths_expr}/{robot_prim_path}',
            name=name,
        )

    def get_v_w(self, indices=None):
        vels = self.get_velocities(indices=indices)
        # return vels[:, 0:2].norm(dim=1), vels[:, 5]
        return vels[:, 0:3].norm(dim=1), vels[:, 3:6].norm(dim=1)


class PumbaaBaseLinkView(XFormPrimView):

    def __init__(
            self,
            base_prim_paths_expr: str,
            name: str = "PumbaaBaskLinkView"
    ):
        self._name = name
        super().__init__(
            prim_paths_expr=f"{base_prim_paths_expr}/{baselink_prim_path}",
            name=name,

        )
