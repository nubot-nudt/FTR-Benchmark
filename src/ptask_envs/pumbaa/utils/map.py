
import traceback

try:
    import omni
    from omni.isaac.occupancy_map import _occupancy_map
except:
    pass

try:
    import cv2
except:
    pass

import numpy as np
from math import floor

from loguru import logger

class MapHelper():

    def __init__(self, lower, upper, cell_size=0.05, path=None, rebuild=True):

        self.lower = np.array(lower)
        self.upper = np.array(upper)
        self.cell_size = cell_size
        self.path = path

        self.compensation = -(np.array(self.lower[:2] / self.cell_size).astype(np.int32))

        if path is None or rebuild == True:
            self.map = np.ones(((self.upper[:2] - self.lower[:2]) / self.cell_size).astype(np.int32)) * -1
        else:
            with open(path, 'rb') as f:
                self.map = np.load(f, allow_pickle=True)

    def save(self, path=None):
        if path is None:
            path = self.path

        with open(path, 'wb') as f:
            self.map.dump(f)

    def get_obs(self, positon, angle, size):
        '''

        :param positon: 裁剪的世界坐标中心点
        :param angle: 旋转角度
        :param size_: 观测范围大小
        :return: 裁剪后高程图
        '''

        size_ = np.array(size)

        pos = np.floor(positon[:2] / self.cell_size + self.compensation)
        side_length = np.sqrt(size_[0] ** 2 + size_[1] ** 2) // self.cell_size

        low = (pos[:2] - side_length // 2)
        up = (pos[:2] + side_length // 2)

        local_map = self.map[int(low[0]):int(up[0])+1, int(low[1]):int(up[1])+1]

        h, w = local_map.shape[0], local_map.shape[1]
        center = (w // 2, h // 2)

        try:
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            rotated = cv2.warpAffine(local_map, M, (w, h))
        except Exception as e:
            logger.error(f'{e}')
            traceback.print_exc()
            return None

        low_clip = np.array(center) - size_ / 2 // self.cell_size
        up_clip = np.array(center) + size_ / 2 // self.cell_size

        return rotated[
            int(low_clip[0]):int(up_clip[0])+1,
            int(low_clip[1]):int(up_clip[1])+1

        ]


    def get_range_map(self, low, up):

        x1 = floor(low[0] / self.cell_size + self.compensation[0])
        y1 = floor(low[1] / self.cell_size + self.compensation[1])

        x2 = floor(up[0] / self.cell_size + self.compensation[0])
        y2 = floor(up[1] / self.cell_size + self.compensation[1])

        return self.map[x1:x2+1, y1:y2+1]


    def compute_map(self):
        for point in self.get_occupied_positions():
            # print(point)
            x = floor(point[0] / self.cell_size + self.compensation[0])
            y = floor(point[1] / self.cell_size + self.compensation[1])
            self.map[x, y] = max(self.map[x, y], point[2])

    def get_occupied_positions(self):

        physx = omni.physx.acquire_physx_interface()
        stage_id = omni.usd.get_context().get_stage_id()

        generator = _occupancy_map.Generator(physx, stage_id)
        generator.update_settings(self.cell_size, 4, 5, 6)
        generator.set_transform((0, 0, 0), self.lower, self.upper)

        logger.info(f'start generator 3d occupied positions')
        generator.generate3d()
        logger.info(f'end generator 3d occupied positions')
        return generator.get_occupied_positions()
