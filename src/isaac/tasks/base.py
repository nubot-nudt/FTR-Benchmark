
from gym import spaces

from omni.isaac.core.tasks.base_task import BaseTask
from omni.isaac.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.viewports import set_camera_view
import torch
import torchvision.transforms.functional as F
from reloading import reloading

from processing.control.flipper import FlipperControl

from isaacsim_ext.prim import add_usd, update_collision

from pumbaa.helper.geometry import Geometryhelper
from pumbaa.helper.map import MapHelper
from pumbaa.sensor.contact import *
from pumbaa.sensor.imu import IMU
from pumbaa.sensor.camera import Camera
from pumbaa.common import AssetEntry
from pumbaa.view.robot import *
from pumbaa.view.articulation import *
from pumbaa.view.rigid import *

class PumbaaBaseTask(BaseTask):

    def __init__(self, name, asset: AssetEntry, offset=None):
        self.num_envs = 1
        self._num_observations = 10 * 32 * 3
        self._num_actions = 6
        self.action_space = spaces.Box(np.array([-0.25, -0.25, -60, -60, -60, -60]), -np.array([-0.25, -0.25, -60, -60, -60, -60]))
        self.observation_space = spaces.Box(
            np.ones(self._num_observations) * -np.Inf, np.ones(self._num_observations) * np.Inf
        )
        self.asset = asset
        self.pumbaa_usd = asset.robot
        try:
            self.map_helper = MapHelper(**asset.map, rebuild=False)
        except FileNotFoundError:
            self.map_helper = MapHelper(**asset.map, rebuild=True)
        try:
            self.geometry_helper = Geometryhelper(**asset.geometry)
        except:
            pass

        self.extractor = torch.nn.AvgPool2d(3)
        self._flipper_control = FlipperControl(self.num_envs, 'cpu')
        BaseTask.__init__(self, name=name, offset=offset)


    @reloading(every=10)
    def get_observations(self) -> dict:
        pos, orient = self.base_link_view.get_world_poses()

        obs = self.map_helper.get_obs(pos[0], np.array(quat_to_euler_angles(orient[0], degrees=True))[2], (2.25, 1.05))
        ext_map = self.extractor(torch.reshape(torch.from_numpy(obs).to('cpu'), (1, 1, 45, 21)))
        ext_map -= pos[0][2] - wheel_radius

        noise = torch.normal(mean=0, std=0.005, size=ext_map.size(), device=ext_map.device)

        v, w = self.robot_view.get_v_w()

        vels = self.robot_view.get_velocities()

        flipper = self.articulation_view.get_all_flipper_positions()

        obs = torch.from_numpy(obs)

        return {
            'camera_img': self.camera.get_image(),
            'img': obs,
            'map': ext_map + noise,
            'pos': pos[0],
            'orient': torch.from_numpy(quat_to_euler_angles(orient[0])),
            'v': v[0],
            'w': w[0],
            'vels': vels[0],
            'flipper': flipper[0],
            'baselink_forces': self.baselink_contact.get_net_contact_forces(),
            'right_and_left_velocities': self.articulation_view.get_right_and_left_velocities(),
            'flipper_velocities': self.articulation_view.get_all_flipper_joint_velocities(),
        }

    def calculate_metrics(self) -> dict:
        pass

    def is_done(self) -> bool:
        pos, orient = self.base_link_view.get_world_poses()

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
        self.articulation_view.set_all_flipper_positions(self._flipper_control.positions)
        self.articulation_view.set_right_and_left_velocities(torch.zeros((1, 2)))
        self.robot_view.set_world_poses(torch.tensor([self.pumbaa_usd.position]), torch.tensor([self.pumbaa_usd.orient]))
        # self.base_link_view.set_world_poses(
        #     torch.tensor([self.pumbaa_usd.position]), torch.tensor([self.pumbaa_usd.orient])
        # )

    def pre_physics_step(self, actions):

        if actions is None:
            return

        if 'vel_type' in actions:
            vel_type = actions['vel_type']
            vels = actions.get('vels', 0)
            if not isinstance(vels, torch.Tensor):
                vels = torch.tensor(vels)

            if vel_type == 'diff':
                self.articulation_view.set_right_and_left_velocities(vels.unsqueeze(dim=0))

            elif vel_type == 'std':
                self.articulation_view.set_v_w(vels.unsqueeze(dim=0))

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

        self.articulation_view.set_all_flipper_position_targets(self._flipper_control.positions)

        pos, orient = self.base_link_view.get_world_poses()
        # self.set_camera_follow(np.array(pos[0]))

    def post_reset(self) -> None:
        self.articulation_view.find_idx()
        # self.wheel_contact.init_contact_sensor()
        self.baselink_contact.initialize()
        self.flipper_contact.initialize()
        self.flipper_contact_2.initialize()
        self.camera.initialize()


        if hasattr(self, 'geometry_helper'):
            try:
                self.geometry_helper.draw()
            except NameError:
                print('could not find _debug_draw')

    def set_up_scene(self, scene: Scene) -> None:
        self._scene = scene

        collision_filters = []

        if self.asset.is_add_ground_plane:
            scene.add_default_ground_plane(prim_path='/ground')
            collision_filters.append('/ground')

        add_usd(self.pumbaa_usd)

        self.base_prim_path = self.pumbaa_usd.prim_path
        self.robot_view = PumbaaRobotView(f"{self.base_prim_path}", name="pumbaa_robot")
        self._scene.add(self.robot_view)

        self.base_link_view = PumbaaBaseLinkView(f"{self.base_prim_path}")

        self.articulation_view = PumbaaArticulationView(f"{self.base_prim_path}")
        scene.add(self.articulation_view)


        for name, usd in self.asset.obstacles.items():
            add_usd(usd)
            collision_filters.append(usd.prim_path)
        for name, terrain in self.asset.terrains.items():
            p = terrain.add_terrain_to_stage(scene.stage)
            collision_filters.append(p)

        self.imu = IMU(self.base_prim_path, scene)
        self.camera = Camera(self.base_prim_path)
        self.baselink_contact = PumbaaBaselinkContact(f'{self.base_prim_path}', collision_filters)
        self.flipper_contact = PumbaaFlipperContact(f'{self.base_prim_path}', collision_filters)
        self.flipper_contact_2 = PumbaaAllFipperContact(f'{self.base_prim_path}', collision_filters)

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




