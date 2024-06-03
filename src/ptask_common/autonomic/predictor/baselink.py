import torch

from ptask_common.processing.perception.height_map import HeightMapConvertor
from ptask_common.processing.robot_info.base import RobotState
from ptask_envs.pumbaa.common.default import robot_length, robot_width


class LSFPredictor():

    def __init__(self, shape=(15, 7), scale=(2.25, 1.05)):
        self.map_convertor = HeightMapConvertor(shape, scale)

    def get_all_points(self, img):
        return self.map_convertor.convert_all_to_points(img)

    def calc_plane(self, points):
        X = torch.cat([points[:, :2], torch.ones((len(points), 1))], dim=1)

        X = torch.autograd.Variable(X)
        y = torch.autograd.Variable(points[:, 2])

        coef, _, _, _ = torch.linalg.lstsq(X, y.view(-1, 1))

        a, b, d = coef[:3].view(-1)
        return a, b, -1, d

    def predict(self, state: RobotState, degree=True):
        points = self.get_all_points(state.height_map)

        robot_points = points[
            (points[:, 0] >= -robot_length / 2) & (points[:, 0] <= robot_length / 2) &
            (points[:, 0] >= -robot_width / 2) & (points[:, 0] <= robot_width / 2)
            ]

        a, b, c, d = self.calc_plane(robot_points)
