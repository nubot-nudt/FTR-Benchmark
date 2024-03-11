
import torch
from processing.robot_info.velocity import world_velocity_to_v, world_velocity_to_w

class RobotState:

    def __init__(self, obs_dict):
        self.height_map = obs_dict.get('map', None).squeeze()
        self.pos = obs_dict.get('pos', None)
        self.orient = obs_dict.get('orient', None)
        self.vels = obs_dict.get('vels', None)
        self.flippers = obs_dict.get('flipper', None)

        self._v = obs_dict.get('v', None)
        self._w = obs_dict.get('w', None)

        self.obs_dict = obs_dict
    def __getitem__(self, item):
        return self.obs_dict[item]

    def __getattr__(self, name):
        return self.obs_dict[name]

    @property
    def vel_roll(self):
        return self.vel_ang_yz

    @property
    def vel_yaw(self):
        return self.vel_ang_xy

    @property
    def vel_pitch(self):
        return self.vel_ang_zx

    @property
    def vel_ang_yz(self):
        return self.vels[3]

    @property
    def vel_ang_zx(self):
        return self.vels[4]

    @property
    def vel_ang_xy(self):
        return self.vels[5]

    @property
    def vel_lin_x(self):
        return self.vels[0]

    @property
    def vel_lin_y(self):
        return self.vels[1]

    @property
    def vel_lin_z(self):
        return self.vels[2]

    @property
    def v(self):
        if self._v is not None:
            return self._v

        return world_velocity_to_v(self.vels)
    @property
    def w(self):
        if self._w is not None:
            return self._w

        return world_velocity_to_w(self.vels)

    @property
    def map(self):
        return self.height_map

    @property
    def pitch(self):
        return self.orient[1]

    @property
    def roll(self):
        return self.orient[0]

    @property
    def yaw(self):
        return self.orient[2]

