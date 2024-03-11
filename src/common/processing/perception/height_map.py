from itertools import product

import torch

class HeightMapConvertor:
    def __init__(self, shape, scale):
        self.shape = shape
        self.center = torch.LongTensor((shape[0] // 2, shape[1] // 2))
        self.ceil_size = torch.FloatTensor((scale[0] / shape[0], scale[1] / shape[1]))

    def get_points_height(self, img, p):
        p = p.reshape(-1, 2)
        x1 = torch.clip(p[:, 0] / self.ceil_size[0] + self.center[0], 0, self.shape[0]-1)
        y1 = torch.clip(p[:, 1] / self.ceil_size[1] + self.center[1], 0, self.shape[1]-1)

        return img.view(self.shape[0], self.shape[1])[torch.floor(x1).long(), torch.floor(y1).long()]

    def get_all_points(self, img):
        points = []
        img = img.view(self.shape[0], self.shape[1])
        for i, j in product(range(self.shape[0]), range(self.shape[1])):
            x = (i - self.center[0]) * self.ceil_size[0]
            y = (j - self.center[1]) * self.ceil_size[1]
            z = img[i, j]
            points.append(torch.stack([x, y, z]))
        return torch.stack(points)


