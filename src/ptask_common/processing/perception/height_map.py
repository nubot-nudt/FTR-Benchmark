from itertools import product

import torch
from ptask_common.utils.tensor import to_tensor, to_same_type


class HeightMapConvertor:
    def __init__(self, shape, scale):
        '''
        转化局部高高程图，地图中心是坐标原点
        :param shape: 地图数组的大小
        :param scale: 地图实际的长宽
        '''
        self.shape = shape
        self.center = torch.LongTensor((shape[0] // 2, shape[1] // 2))
        self.ceil_size = torch.FloatTensor((scale[0] / shape[0], scale[1] / shape[1]))

    def convert_point_to_index(self, p):
        x1 = torch.clip(p[0] / self.ceil_size[0] + self.center[0], 0, self.shape[0] - 1)
        y1 = torch.clip(p[1] / self.ceil_size[1] + self.center[1], 0, self.shape[1] - 1)
        return [x1, y1]

    def get_points_heights(self, img, p):
        p = p.reshape(-1, 2)
        x1 = torch.clip(p[:, 0] / self.ceil_size[0] + self.center[0], 0, self.shape[0] - 1)
        y1 = torch.clip(p[:, 1] / self.ceil_size[1] + self.center[1], 0, self.shape[1] - 1)

        return img.view(self.shape[0], self.shape[1])[torch.floor(x1).long(), torch.floor(y1).long()]

    def convert_all_to_points(self, img):
        points = []
        img = img.view(self.shape[0], self.shape[1])
        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            x = (i - self.center[0]) * self.ceil_size[0]
            y = (j - self.center[1]) * self.ceil_size[1]
            z = img[i, j]
            points.append(torch.stack([x, y, z]))
        return torch.stack(points)


class BigMapConvertor():
    def __init__(self, height_map_array, lower, upper, cell_size):
        self.height_map_array = to_tensor(height_map_array)
        self.lower = to_tensor(lower)
        self.upper = to_tensor(upper)
        self.cell_size = cell_size

        self.compensation = -(self.lower[:2] / self.cell_size)

    def get_heights(self, p):
        p = p[:, :2] / self.cell_size + self.compensation
        i = p.long()
        return self.height_map_array[i[:, 0], i[:, 1]]

    def get_world_point_height(self, _p):
        p = to_tensor(_p)
        p_ = p[:2] / self.cell_size + self.compensation
        index = p_.long()
        return self.height_map_array[index[0], index[1]].item()

    def convert_world_point_to_array_index(self, _p):
        p = to_tensor(_p)
        p_ = p[:2] / self.cell_size + self.compensation
        return to_same_type(p_, _p)

    def convert_array_index_to_world_point(self, _p):
        p = to_tensor(_p)
        p_ = (p - self.compensation) * self.cell_size
        return to_same_type(p_, _p)
