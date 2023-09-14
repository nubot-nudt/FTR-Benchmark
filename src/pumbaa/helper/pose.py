import torch

from ..common.default import flipper_end_point, flipper_start_point, robot_length, robot_width

class RobotRecommandPoseHelper():

    def __init__(self, shape, scale):
        self.shape = shape
        self.center = torch.LongTensor((shape[0] // 2, shape[1] // 2))
        self.ceil_size = torch.FloatTensor((scale[0] / shape[0], scale[1] / shape[1]))
        self.flipper_start_point = torch.tensor(flipper_start_point)

        self.count = 5
        self.flipper_points = torch.zeros((len(flipper_end_point), self.count, 2))
        self.flipper_distances = torch.zeros((len(flipper_end_point), self.count))
        for i, s, e in zip(range(len(flipper_end_point)), flipper_start_point, flipper_end_point):
            self.flipper_points[i, :, 0] = torch.linspace(s[0], e[0], self.count)
            self.flipper_points[i, :, 1] = torch.linspace(s[1], e[1], self.count)

            self.flipper_distances[i, :] = torch.sqrt(
                (self.flipper_points[i, :, 0] - flipper_start_point[i][0]) ** 2 \
                + (self.flipper_points[i, :, 1] - flipper_start_point[i][1]) ** 2
            )

        self.robot_points = torch.cartesian_prod(
            torch.linspace(-robot_length / 2, robot_length / 2, self.count * 2),
            torch.linspace(-robot_width / 2, robot_width / 2, self.count * 2),
        )

    def get_point_height(self, img, p):
        p = p.view(-1, 2)
        x1 = p[:, 0] / self.ceil_size[0] + self.center[0]
        y1 = p[:, 1] / self.ceil_size[1] + self.center[1]
        return img.view(self.shape[0], self.shape[1])[torch.floor(x1).long(), torch.floor(y1).long()]

    def calc_flipper_position(self, img):
        flipper_start_heights = self.get_point_height(img, self.flipper_start_point).view(4, -1).expand(4, 5)
        flipper_end_heights = self.get_point_height(img, self.flipper_points).view(len(flipper_end_point), self.count)

        heights = flipper_end_heights - flipper_start_heights

        theta = torch.rad2deg(
            torch.arctan(
                heights / self.flipper_distances
            )
        )

        return theta[:, 1:]

    def calc_robot_orient(self, img):
        heights = self.get_point_height(img, self.robot_points)
        X = torch.cat([self.robot_points, heights.view(-1, 1)], dim=1)
        w, _, _, _ = torch.linalg.lstsq(X, torch.ones_like(heights))
        norm = w.norm()
        pitch = torch.arccos(w[2] / norm)
        roll = torch.arccos(w[1] / norm)

        return torch.abs(90 - torch.rad2deg(pitch)), torch.abs(90 - torch.rad2deg(roll))