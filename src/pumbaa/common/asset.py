import os
import typing

import yaml
from loguru import logger

_prim_path_pool = set()
_matched_path = set()

class UsdPrimInfo:
    def __init__(
            self,
             path,
             position: typing.Sequence[float] = (0, 0, 0),
             orient: typing.Sequence[float] = (1, 0, 0, 0),
             scale: typing.Sequence[float] = (1, 1, 1),
    ):

        self.path = path
        self.position = position
        self.orient = orient
        self.scale =scale

        if len(set(self.path) & set('*[]%')) > 0:
            try:
                self.path = os.popen(f'ls -t1 {self.path}').read().strip().split('\n')[0]
            except IndexError:
                raise FileNotFoundError(f'{path} don\'t exist')

            if path not in _matched_path:
                logger.info(f'{path} -> {self.path}')

            _matched_path.add(path)

    @property
    def prim_path(self) -> str:
        '''
        获取在/World下默认prim路经
        :return:
        '''
        if hasattr(self, '_prim_path'):
            return self._prim_path

        path = f'/World/{self.name}'
        if path not in _prim_path_pool:
            _prim_path_pool.add(path)
            self._prim_path = path
            return self._prim_path

        for i in range(1, 100):
            n_path = f'{path}_{i}'
            if n_path not in _prim_path_pool:
                _prim_path_pool.add(n_path)
                self._prim_path = n_path
                return self._prim_path
    @property
    def name(self) -> str:
        '''
        获取prim的默认名字，根据文件名生成，在实际使用中不一定使用到
        :return:
        '''
        t = os.path.basename(self.path).replace('.usd', '').replace(".", "_").replace('-', '_')
        if t[0].isnumeric():
            return 'v' + t
        return t


class AssetEntry():
    def __init__(self, name='', file=None):
        self.name: str = name
        self.robot: UsdPrimInfo = None
        self.obstacles: typing.Dict[str, UsdPrimInfo] = {}
        self.map: dict = None
        self.geometry: dict = None
        self.scene: dict = None
        self.camera: dict = None

        if file is not None:
            self.load(file)

    def load(self, file):
        with open(file, 'r') as f:
            yaml_info = yaml.safe_load(f)
        self.name = yaml_info['name']
        self.set_robot(UsdPrimInfo(**yaml_info['robot']))

        field_name = ['path', 'position', 'orient', 'scale']
        root_obstacle_field = {k:v for k,v in yaml_info['obstacle'].items() if k in field_name}

        if 'path' in root_obstacle_field:
            self.add_obstacle('default', UsdPrimInfo(**root_obstacle_field))

        for k,v in yaml_info['obstacle'].items():
            if k in field_name:
                continue
            self.add_obstacle(k, UsdPrimInfo(**v))

        self.map = yaml_info['map']
        self.geometry = yaml_info.get('geometry', None)

        if 'scene' in yaml_info:
            self.scene = yaml_info['scene']

        if 'camera' in yaml_info:
            self.camera = yaml_info['camera']

    def set_robot(self, robot_usd: UsdPrimInfo=None):
        if robot_usd == None:
            self.robot = UsdPrimInfo('')
        else:
            self.robot = robot_usd

    def add_obstacle(self, name, obstacle_usd: UsdPrimInfo):
        self.obstacles[name] = obstacle_usd

    def has_geometry(self):
        return not self.geometry is None

    def has_camera(self):
        return not self.camera is None

    @property
    def is_add_ground_plane(self) -> bool:
        if self.scene is None:
            return False
        return self.scene.get('add_ground_plane', False)