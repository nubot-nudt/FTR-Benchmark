from typing import Optional
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.articulations import ArticulationView
from ptask_envs.pumbaa.utils.prim import *
from omni.isaac.core.utils.torch.rotations import *
from omni.isaac.core.prims import RigidPrimView
from pxr import PhysxSchema

from ptask_envs.pumbaa.common.default import *


class PumbaaRobot(Robot):
    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "PumbaaWheel",
            # usd_path: Optional[str] = None,
            translation: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._name = name
        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

    def set_robot_env_config(self, pum_cfg):

        for joint_name in flipper_joint_names:
            joint_path = f'{self.prim_path}/chassis_link/{joint_name}'
            set_joint_max_vel(joint_path, pum_cfg.get('flipper_joint_max_vel', 20))
            set_joint_stiffness(joint_path, pum_cfg.get('flipper_joint_stiffness', 1000))
            set_joint_damping(joint_path, pum_cfg.get('flipper_joint_damping', 0))

        for chassis_base in find_matching_prim_paths(f'{self.prim_path}/wheel_list/wheel_*/[LR]*'):
            chassis_render = f'{chassis_base}/Render'
            chassis_joint = f'{chassis_base}/{os.path.basename(chassis_base)}RevoluteJoint'
            get_prim_at_path(chassis_render) \
                .GetAttribute('physics:mass') \
                .Set(pum_cfg.get('chassis_wheel_render_mass', 3))
            set_joint_stiffness(chassis_joint, pum_cfg.get('chassis_wheel_joint_stiffness', 0))
            set_joint_damping(chassis_joint, pum_cfg.get('chassis_wheel_joint_damping', 1000))

        for flipper_base in find_matching_prim_paths(f'{self.prim_path}/flipper_list/[fr]*/[FR]*'):
            flipper_render = f'{flipper_base}/FlipperRender'
            get_prim_at_path(flipper_render) \
                .GetAttribute('physics:mass') \
                .Set(pum_cfg.get('flipper_wheel_render_mass', 3))

        for flipper_wheel_joint in find_matching_prim_paths(f'{self.prim_path}/flipper_list/[fr]*/[FR]*/.*'):
            if 'FlipperRender' in flipper_wheel_joint:
                continue
            set_joint_stiffness(flipper_wheel_joint, pum_cfg.get('flipper_wheel_joint_stiffness', 0))
            set_joint_damping(flipper_wheel_joint, pum_cfg.get('flipper_wheel_joint_damping', 1000))

        set_material_friction(f'{self.prim_path}/Looks/flipper_material', pum_cfg.get('flipper_material_friction', 1))
        set_material_friction(f'{self.prim_path}/Looks/wheel_material', pum_cfg.get('wheel_material_friction', 1))

    def set_pumbaa_default_properties(self, stage):
        for link_prim in get_prim_at_path(f'{self.prim_path}').GetChildren():
            if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
                rb.GetDisableGravityAttr().Set(False)
                rb.GetRetainAccelerationsAttr().Set(False)
                rb.GetLinearDampingAttr().Set(0.0)
                rb.GetMaxLinearVelocityAttr().Set(1000.0)
                rb.GetAngularDampingAttr().Set(0.0)
                rb.GetMaxAngularVelocityAttr().Set(64 / np.pi * 180)

    def prepare_contacts(self, stage):
        # for link_prim in get_prim_at_path(f'{self.prim_path}').GetChildren():
        #     if link_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        #         if "_HIP" not in str(link_prim.GetPrimPath()):
        #             rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, link_prim.GetPrimPath())
        #             rb.CreateSleepThresholdAttr().Set(0)
        #             cr_api = PhysxSchema.PhysxContactReportAPI.Apply(link_prim)
        #             cr_api.CreateThresholdAttr().Set(0)

        for link_prim in chain(
                find_matching_prim_paths(f'{self.prim_path}/wheel_list/wheel_*/[LR]*'),
                find_matching_prim_paths(f'{self.prim_path}/flipper_list/[fr]*/[FR]*')
        ):
            prim = get_prim_at_path(link_prim)
            if prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
                rb = PhysxSchema.PhysxRigidBodyAPI.Get(stage, prim.GetPrimPath())
                rb.CreateSleepThresholdAttr().Set(0)
                cr_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
                cr_api.CreateThresholdAttr().Set(0)


class PumbaaRobotView(ArticulationView):
    def __init__(
            self,
            prim_paths_expr: str,
            name: Optional[str] = "PumbaaRobotView",
            track_contact_forces=False,
            prepare_contact_sensors=False,
    ):
        super().__init__(prim_paths_expr=prim_paths_expr, name=name, reset_xform_properties=False)
        self.chassis_wheel = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/.*/pumbaa_wheel/wheel_list/wheel_*/[LR]*",
            name="chassis_wheel_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )
        self.flipper_wheel = RigidPrimView(
            prim_paths_expr=f"/World/envs/.*/.*/pumbaa_wheel/flipper_list/[fr]*/[FR]*",
            name="flipper_wheel_view",
            reset_xform_properties=False,
            track_contact_forces=track_contact_forces,
            prepare_contact_sensors=prepare_contact_sensors,
        )

    def get_all_flipper_efforts(self, indices=None):
        return self.get_applied_joint_efforts(indices=indices, joint_indices=self.flipper_dof_idx_list)

    def get_all_flipper_joint_velocities(self, indices=None):
        return self.get_joint_velocities(indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_all_flipper_dt(self, dt_positions: torch.Tensor, indices=None, max_position=60 / 180 * torch.pi,
                           degree=False):
        current_pos = self.get_all_flipper_positions(indices=indices, degree=degree)
        a = torch.tensor([-1, -1, 1, 1], device=dt_positions.device)
        pos = current_pos + dt_positions
        pos = torch.clip(pos, -max_position, max_position)

        pos = torch.deg2rad(pos * a) if degree else pos * a

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def get_all_flipper_positions(self, indices=None, degree=False):
        positions = self.get_joint_positions(indices=indices, joint_indices=self.flipper_dof_idx_list)

        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        return torch.rad2deg(positions * a) if degree else positions * a

    def set_all_flipper_positions_with_dt(self, positions: torch.Tensor, dt, indices=None, degree=False):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = positions

        current_pos = self.get_all_flipper_positions(indices=indices)
        for i in range(len(current_pos)):
            for j in range(4):
                if current_pos[i][j] > pos[i][j]:
                    pos[i][j] = torch.max(pos[i][j], current_pos[i][j] - dt)
                else:
                    pos[i][j] = torch.min(pos[i][j], current_pos[i][j] + dt)

        pos = torch.deg2rad(pos * a) if degree else pos * a

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_all_flipper_position_targets(self, positions: torch.Tensor, indices=None, degree=False):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)

        pos = torch.deg2rad(positions * a) if degree else positions * a

        self.set_joint_position_targets(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_all_flipper_positions(self, positions: torch.Tensor, indices=None, degree=False):
        a = torch.tensor([-1, -1, 1, 1], device=positions.device)
        pos = torch.deg2rad(positions * a) if degree else positions * a

        self.set_joint_positions(pos, indices=indices, joint_indices=self.flipper_dof_idx_list)

    def set_v_w(self, v_w: torch.Tensor, indices=None):
        v = v_w[:, 0]
        w = v_w[:, 1]

        vels = torch.zeros(v_w.shape, device=v_w.device)
        vels[:, 0] = (2 * v + w * L) / 2
        vels[:, 1] = (2 * v - w * L) / 2

        return self.set_right_and_left_velocities(vels, indices=indices)

    def get_v_w(self, indices=None):
        v_r, v_l = self.get_right_and_left_velocities(indices=indices)

        return (v_r + v_l) / 2, (v_r - v_l) / L

    def set_right_and_left_velocities(self, vels, indices=None):
        # r_v = torch.ones((vels.shape[0], len(self.r_indices)), device=vels.device)
        # l_v = torch.ones((vels.shape[0], len(self.l_indices)), device=vels.device)
        # fr_v = torch.ones((vels.shape[0], len(self.fr_indices)), device=vels.device)
        # fl_v = torch.ones((vels.shape[0], len(self.fl_indices)), device=vels.device)
        #
        # for i in range(vels.shape[0]):
        #     r_v[i, :] *= -vels[i, 0] / wheel_radius
        #     l_v[i, :] *= -vels[i, 1] / wheel_radius
        #     fr_v[i, :] *= vels[i, 0] / wheel_radius
        #     fl_v[i, :] *= vels[i, 1] / wheel_radius
        #
        # self.set_joint_velocities(r_v, indices=indices, joint_indices=self.r_indices)
        # self.set_joint_velocities(l_v, indices=indices, joint_indices=self.l_indices)
        # self.set_joint_velocities(fr_v, indices=indices, joint_indices=self.fr_indices)
        # self.set_joint_velocities(fl_v, indices=indices, joint_indices=self.fl_indices)

        self.set_joint_velocities((-vels[:, 0] / wheel_radius).unsqueeze(dim=-1).repeat(1, len(self.r_indices)), indices=indices, joint_indices=self.r_indices)
        self.set_joint_velocities((-vels[:, 1] / wheel_radius).unsqueeze(dim=-1).repeat(1, len(self.l_indices)), indices=indices, joint_indices=self.l_indices)
        self.set_joint_velocities((vels[:, 0] / wheel_radius).unsqueeze(dim=-1).repeat(1, len(self.fr_indices)), indices=indices, joint_indices=self.fr_indices)
        self.set_joint_velocities((vels[:, 1] / wheel_radius).unsqueeze(dim=-1).repeat(1, len(self.fl_indices)), indices=indices, joint_indices=self.fl_indices)

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
