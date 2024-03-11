import torch


from processing.perception.height_map import HeightMapConvertor

from ..common.default import flipper_end_point, flipper_start_point, robot_length, robot_width


class RobotRecommandPoseHelper():

    def __init__(self, shape, scale):
        self.map_convertor = HeightMapConvertor(shape, scale)
        self.flipper_start_point = torch.tensor(flipper_start_point)

        self.count = 5
        self.flipper_points = torch.zeros((len(flipper_end_point), self.count, 2))
        self.flipper_distances = torch.zeros((len(flipper_end_point), self.count))
        offset = [0.0, 0.0, 0, 0]
        self.set_offset(offset)

        self.robot_points = torch.cartesian_prod(
            torch.linspace(-robot_length / 2, robot_length / 2, self.count * 2),
            torch.linspace(-robot_width / 2, robot_width / 2, self.count * 2),
        )


    def set_offset(self, offset):
        '''

        :param offset: 预测摆臂位置的偏移值
        :return:
        '''
        for i, s, e in zip(range(len(flipper_end_point)), flipper_start_point, flipper_end_point):
            self.flipper_points[i, :, 0] = torch.linspace(s[0] + offset[i], e[0] + offset[i], self.count)
            self.flipper_points[i, :, 1] = torch.linspace(s[1] + offset[i], e[1] + offset[i], self.count)

            self.flipper_distances[i, :] = torch.sqrt(
                (self.flipper_points[i, :, 0] - flipper_start_point[i][0]) ** 2 \
                + (self.flipper_points[i, :, 1] - flipper_start_point[i][1]) ** 2
            )

    def get_point_height(self, img, p):
        return self.map_convertor.get_points_height(img, p)

    def calc_flipper_except_position(self, img, roll, pitch):
        T_roll_matrix = torch.DoubleTensor([
            [1, 0, 0],
            [0, torch.cos(roll), -torch.sin(roll)],
            [0, torch.sin(roll), torch.cos(roll)],
        ])

        T_pitch_matrix = torch.DoubleTensor([
            [torch.cos(pitch), 0, torch.sin(pitch)],
            [0, 1, 0],
            [-torch.sin(pitch), 0, torch.cos(pitch)],
        ])
        p = torch.DoubleTensor([
            (0.28, 0.26, 0),
            (0.28, -0.26, 0),
            (-0.28, 0.26, 0),
            (-0.28, -0.26, 0)
        ])
        points = (T_roll_matrix @ T_pitch_matrix @ p.T).T

        position_num = 3
        # flipper_start_heights = points[:, 2] + robot_height
        flipper_start_heights = points[:, 2] + self.get_point_height(img, torch.LongTensor([0, 0]))

        flipper_end_heights = self.get_point_height(img, self.flipper_points[:, -position_num:, :]).view(len(flipper_end_point), position_num)

        heights = (flipper_end_heights.T - flipper_start_heights).T

        theta = torch.rad2deg(
            torch.arctan(
                heights / self.flipper_distances[:, -position_num:]
            )
        )
        return theta

    def calc_flipper_position(self, img):
        position_num = 3
        flipper_start_heights = self.get_point_height(img, self.flipper_start_point).view(4, -1).expand(4, position_num)

        flipper_end_heights = self.get_point_height(img, self.flipper_points[:, -position_num:, :]).view(len(flipper_end_point), position_num)

        heights = flipper_end_heights - flipper_start_heights

        theta = torch.rad2deg(
            torch.arctan(
                heights / self.flipper_distances[:, -position_num:]
            )
        )

        return theta