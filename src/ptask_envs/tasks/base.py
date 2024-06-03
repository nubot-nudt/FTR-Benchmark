from gym import spaces
import numpy as np
import torch
import math

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.viewports import set_camera_view
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.torch.rotations import *
from reloading import reloading

from ptask_common.processing.control.flipper import FlipperControl
from ptask_envs.pumbaa.common.default import *

# from ptask_envs.pumbaa.helper.geometry import Geometryhelper
from ptask_envs.pumbaa.utils.map import MapHelper
from ptask_envs.pumbaa.sensor.imu import IMU
from ptask_envs.pumbaa.sensor.camera import Camera
from ptask_envs.pumbaa.utils.asset import AssetEntry
from ptask_envs.pumbaa.utils.prim import update_collision, add_usd
from ptask_envs.pumbaa.view.robot import PumbaaRobot, PumbaaRobotView


def euler_from_quaternion(w, x, y, z):
    # 通过四元数计算欧拉角 (roll, pitch, yaw)
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class PumbaaBaseTask(BaseTask):

    def __init__(self, name, asset: AssetEntry, offset=None):
        self.num_envs = 1
        self._num_observations = 10 * 32 * 3
        self._num_actions = 6
        self.action_space = spaces.Box(np.array([-0.25, -0.25, -60, -60, -60, -60]),
                                       -np.array([-0.25, -0.25, -60, -60, -60, -60]))
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        )
        self.asset = asset
        self.pumbaa_usd = asset.robot
        try:
            self.map_helper = MapHelper(**asset.map, rebuild=False)
        except FileNotFoundError:
            self.map_helper = MapHelper(**asset.map, rebuild=True)
        # try:
        #     self.geometry_helper = Geometryhelper(**asset.geometry)
        # except:
        #     pass

        self.extractor = torch.nn.AvgPool2d(3)
        self._flipper_control = FlipperControl(self.num_envs, 'cpu')
        BaseTask.__init__(self, name=name, offset=offset)

        self.position = torch.zeros((3,))

    @reloading(every=10)
    def get_observations(self) -> dict:
        pos, orient = self.pumbaa_robots.get_world_poses()
        self.position = pos[0]

        obs = self.map_helper.get_obs(pos[0], np.array(quat_to_euler_angles(orient[0], degrees=True))[2], (2.25, 1.05))
        ext_map = self.extractor(torch.reshape(torch.from_numpy(obs).to('cpu'), (1, 1, 45, 21)))
        ext_map -= pos[0][2] - wheel_radius

        noise = torch.normal(mean=0, std=0.005, size=ext_map.size(), device=ext_map.device)
        vels = self.pumbaa_robots.get_velocities()
        flipper = self.pumbaa_robots.get_all_flipper_positions()
        obs = torch.from_numpy(obs)

        return {
            'img': obs,
            'map': ext_map + noise,
            'pos': pos[0],
            'orient': torch.from_numpy(quat_to_euler_angles(orient[0])),
            # 'orient': torch.tensor(euler_from_quaternion(*orient[0])),
            'orient_quat': orient[0],
            'v': quat_rotate_inverse(orient, vels[:, :3])[0, 0],
            'w': quat_rotate_inverse(orient, vels[:, 3:])[0, -1],
            # 'v': vels[0, 0],
            'vels': vels[0],
            'flipper': flipper[0],
            'chassis_forces': self.pumbaa_robots.chassis_wheel.get_net_contact_forces().view(16, 3),
            'flipper_forces': self.pumbaa_robots.flipper_wheel.get_net_contact_forces().view(20, 3),
            'flipper_h': self.pumbaa_robots.flipper_wheel.get_world_poses()
        }

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        pos, orient = self.pumbaa_robots.get_world_poses()

        point = (pos[0][0], pos[0][1])

        if hasattr(self, 'geometry_helper'):

            if not self.geometry_helper.is_in_range(point):
                return True

            if self.geometry_helper.is_in_target(point):
                return True

        return False

    def set_params(self, *args, **kwargs) -> None:
        pass

    def get_params(self) -> dict:
        pass

    def reset(self):
        for name, usd in self.asset.obstacles.items():
            update_collision(usd)

        self._flipper_control.zero()
        self.pumbaa_robots.set_all_flipper_positions(self._flipper_control.positions)
        self.pumbaa_robots.set_right_and_left_velocities(torch.zeros((1, 2)))
        self.pumbaa_robots.set_world_poses(torch.tensor([self.pumbaa_usd.position]),
                                           torch.tensor([self.pumbaa_usd.orient]))

    def set_robot_position(self, pos):
        _, orient = self.pumbaa_robots.get_world_poses()
        self.pumbaa_robots.set_all_flipper_positions(self._flipper_control.positions)
        self.pumbaa_robots.set_right_and_left_velocities(torch.zeros((1, 2)))
        self.pumbaa_robots.set_world_poses(torch.tensor([pos]), orient)

    def pre_physics_step(self, actions):

        if actions is None:
            return

        if 'vel_type' in actions:
            vel_type = actions['vel_type']
            vels = actions.get('vels', 0)
            if not isinstance(vels, torch.Tensor):
                vels = torch.tensor(vels)

            if vel_type == 'diff':
                self.pumbaa_robots.set_right_and_left_velocities(vels.unsqueeze(dim=0))

            elif vel_type == 'std':
                self.pumbaa_robots.set_v_w(vels.unsqueeze(dim=0))

        if 'flipper_type' in actions:
            flipper_type = actions['flipper_type']
            flippers = actions.get('flippers', [0, 0, 0, 0])
            if not isinstance(flippers, torch.Tensor):
                flippers = torch.tensor(flippers)
            flippers = flippers.float()

            if flipper_type == 'pos':
                self._flipper_control.set_pos(flippers.unsqueeze(dim=0))
            elif flipper_type == 'pos_dt':
                self._flipper_control.set_pos_with_dt(flippers.unsqueeze(dim=0))
            elif flipper_type == 'dt':
                self._flipper_control.set_pos_dt(flippers.unsqueeze(dim=0))

        self.pumbaa_robots.set_all_flipper_position_targets(self._flipper_control.positions, degree=True)

        pos, orient = self.pumbaa_robots.get_world_poses()
        # self.set_camera_follow(np.array(pos[0]))

    def post_reset(self) -> None:
        self.pumbaa_robots.find_idx()
        # self.wheel_contact.init_contact_sensor()
        # self.baselink_contact.initialize()
        # self.flipper_contact.initialize()
        # self.flipper_contact_2.initialize()
        self.camera.initialize()

        if hasattr(self, 'geometry_helper'):
            try:
                self.geometry_helper.draw()
            except NameError:
                print('could not find _debug_draw')

    def set_up_scene(self, scene: Scene) -> None:
        self._scene = scene
        self._stage = get_current_stage()

        collision_filters = []

        if self.asset.is_add_ground_plane:
            scene.add_default_ground_plane(prim_path='/ground')
            collision_filters.append('/ground')

        self.base_prim_path = '/World/envs/env_0/' + self.pumbaa_usd.name
        add_usd(self.pumbaa_usd, prim_path=self.base_prim_path)
        self.pumbaa_robot = PumbaaRobot(f"{self.base_prim_path}/{robot_base_name}", name=robot_base_name)
        self.pumbaa_robot.set_pumbaa_default_properties(self._stage)
        self.pumbaa_robot.prepare_contacts(self._stage)

        self._scene.add(self.pumbaa_robot)

        self.pumbaa_robots = PumbaaRobotView(f"{self.base_prim_path}/{robot_base_name}",
                                             track_contact_forces=True,
                                             prepare_contact_sensors=False)
        scene.add(self.pumbaa_robots)
        scene.add(self.pumbaa_robots.chassis_wheel)
        scene.add(self.pumbaa_robots.flipper_wheel)

        for name, usd in self.asset.obstacles.items():
            add_usd(usd)
            collision_filters.append(usd.prim_path)
        for name, terrain in self.asset.terrains.items():
            p = terrain.add_terrain_to_stage(scene.stage)
            collision_filters.append(p)

        self.imu = IMU(self.base_prim_path, scene)
        self.camera = Camera(self.base_prim_path)

        # self.baselink_contact = PumbaaBaselinkContact(f'{self.base_prim_path}', collision_filters)
        # self.flipper_contact = PumbaaFlipperContact(f'{self.base_prim_path}', collision_filters)
        # self.flipper_contact_2 = PumbaaAllFipperContact(f'{self.base_prim_path}', collision_filters)

        # from pumbaa.sensor.imu import IMU
        # self.imu = IMU(self.base_prim_path, scene)

        if self.asset.has_camera():
            self.set_initial_camera_params(**self.asset.camera)

    def set_camera_follow(self, position):
        self.set_initial_camera_params(
            [position[0] + 4, position[1] + 4, position[2] + 4],
            [position[0], position[1] - 1, position[2]]
        )

    def set_initial_camera_params(self, position=[10, 10, 3], target=[0, 0, 0]):
        set_camera_view(eye=position, target=target, camera_prim_path="/OmniverseKit_Persp")
