import torch

from pumbaa.helper.pose import RobotRecommandPoseHelper

from autonomic.common.base import BasePredictor
from processing.robot_info.base import RobotState

class FlipperPredictor(BasePredictor):

    def __init__(self, shape=(15, 7), scale=(2.25, 1.05)):
        self.helper = RobotRecommandPoseHelper(shape=shape, scale=scale)

    def predict(self, state: RobotState, degree=True):

        pos = self.helper.calc_flipper_position(state.map).mean(dim=-1)
        pitch = torch.rad2deg(state.pitch)  # 向上为负
        recommand_pos = pos + pitch * torch.tensor([1, 1, -1, -1])

        # print(state.pos)

        # recommand_pos = self.helper.calc_flipper_except_position(state.map, state.pitch, torch.tensor(0)).mean(dim=-1)


        return recommand_pos if degree else torch.deg2rad(recommand_pos)
