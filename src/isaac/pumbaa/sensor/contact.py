from enum import Enum

import torch


from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.prims import find_matching_prim_paths, get_prim_at_path
from omni.isaac.core.utils.transformations import get_translation_from_target

from pumbaa.common.default import *

__contact_names_i = 0

def _get_unique_name():
    global __contact_names_i
    name = f'ContactSensor_{__contact_names_i}'
    __contact_names_i += 1
    return name

class WheelContactType(Enum):
    BASELINK = 0
    FLIPPER = 1
class WheelContact():

    def __init__(
            self,
            base_prim_path,
            scene: Scene,
            type: WheelContactType = WheelContactType.BASELINK,
    ):
        self.sensors = []
        self.base_prim_path = base_prim_path
        self.type = type
        self._scene = scene

        # self.add_contact_sensor(base_prim_path)


    def add_contact_sensor(self, base_prim_path):

        if self.type == WheelContactType.BASELINK:
            matching_path = f'{base_prim_path}/{baselink_wheel_render_prim_path}'
        elif self.type == WheelContactType.FLIPPER:
            matching_path = f'{base_prim_path}/{flipper_render_prim_path}'

        from omni.isaac.sensor import ContactSensor
        for i, path in enumerate(find_matching_prim_paths(matching_path)):
            name = _get_unique_name()
            sensor = self._scene.add(ContactSensor(prim_path=f'{path}/{name}', name=name))
            self.sensors.append(sensor)

    def init_contact_sensor(self):
        for sensor in self.sensors:
            sensor.add_raw_contact_data_to_frame()

    def get_contact_local_points(self):

        if not hasattr(self, '_is_init'):
            self.init_contact_sensor()
            self._is_init = True

        positions = list()
        for sensor in self.sensors:
            for contact in sensor.get_current_frame()['contacts']:
                position = contact['position']
                position = get_translation_from_target(
                    position.numpy(),
                    source_prim=get_prim_at_path('/World'),
                    target_prim=get_prim_at_path(f'{self.base_prim_path}/{baselink_prim_path}')
                )
                if position[2] > 0:
                    continue
                positions.append(torch.tensor(position))

        return torch.stack(positions, dim=0) if len(positions) > 0 else torch.tensor([])

    def get_contact_points(self):
        if not hasattr(self, '_is_init'):
            self.init_contact_sensor()
            self._is_init = True

        positions = list()
        for sensor in self.sensors:
            for contact in sensor.get_current_frame()['contacts']:
                position = contact['position']
                positions.append(position)

        return torch.stack(positions, dim=0) if len(positions) > 0 else torch.tensor([])