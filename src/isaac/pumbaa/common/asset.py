import os
import typing

import yaml
from loguru import logger
from isaacsim_ext.usd import UsdPrimInfo
from isaacsim_ext.terrain import TerrainInfo

from utils.common.asset import *

class AssetEntry():
    def __init__(self, name='', file=None):
        self.name: str = name
        self.robot: UsdPrimInfo = None
        self.obstacles: typing.Dict[str, UsdPrimInfo] = {}
        self.terrains = dict()
        self.map: dict = None
        self.scene: dict = None
        self.camera: dict = None
        self.task_info: dict = None

        if file is not None:
            self.load(file)

    def load(self, file):
        with open(file, 'r') as f:
            yaml_info = yaml.safe_load(f)

        self.name = os.path.basename(file).removesuffix('.yaml')
        self.set_robot(UsdPrimInfo(**yaml_info['robot']))

        if 'obstacle' in yaml_info:
            field_name = ['path', 'position', 'orient', 'scale']
            root_obstacle_field = {k:v for k,v in yaml_info['obstacle'].items() if k in field_name}

            if 'path' in root_obstacle_field:
                self.add_obstacle('default', UsdPrimInfo(**root_obstacle_field))

            for k,v in yaml_info['obstacle'].items():
                if k in field_name:
                    continue
                self.add_obstacle(k, UsdPrimInfo(**v))

        self.map = yaml_info['map']

        self.map['path'] = get_map_file_path_by_name(self.name)

        if 'terrain' in yaml_info:
            for k, v in yaml_info['terrain'].items():
                self.add_terrain(k, v)

        if 'scene' in yaml_info:
            self.scene = yaml_info['scene']

        if 'camera' in yaml_info:
            self.camera = yaml_info['camera']

        if 'task_info' in yaml_info:
            self.task_info = yaml_info['task_info']

    def set_robot(self, robot_usd: UsdPrimInfo=None):
        if robot_usd == None:
            self.robot = UsdPrimInfo('')
        else:
            self.robot = robot_usd

    def add_obstacle(self, name, obstacle_usd: UsdPrimInfo):
        self.obstacles[name] = obstacle_usd

    def add_terrain(self, name, terrain_cfg):
        terrain_list = terrain_cfg.get('terrain_list', [])

        if len(terrain_list) == 0:
            return

        self.terrains[name] = TerrainInfo(terrain_cfg)

    def has_camera(self):
        return self.camera is not None

    def has_task_info(self):
        return self.task_info is not None

    @property
    def is_add_ground_plane(self) -> bool:
        if self.scene is None:
            return False
        return self.scene.get('add_ground_plane', False)