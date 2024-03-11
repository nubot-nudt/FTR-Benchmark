import typing
import os

from utils.log import logger

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
        self.scale = scale

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
