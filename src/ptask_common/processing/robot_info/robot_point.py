import torch

from ptask_envs.pumbaa.common.default import flipper_length as _flipper_length
from ptask_envs.pumbaa.common.default import flipper_joint_x
from ptask_envs.pumbaa.common.default import flipper_joint_y


def robot_to_world_matrix(roll, pitch, dtype=torch.float64):
    T_roll_matrix = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(roll), -torch.sin(roll)],
        [0, torch.sin(roll), torch.cos(roll)],
    ], dtype=dtype)

    T_pitch_matrix = torch.tensor([
        [torch.cos(pitch), 0, torch.sin(pitch)],
        [0, 1, 0],
        [-torch.sin(pitch), 0, torch.cos(pitch)],
    ], dtype=dtype)

    return T_roll_matrix @ T_pitch_matrix

def robot_to_world(roll, pitch, points):
    return (robot_to_world_matrix(roll, pitch, dtype=points.dtype) @ points.T).T


def robot_flipper_positions(flipper_angle, degree=False, flipper_length=_flipper_length):
    '''

    :param degree:
    :param flipper_length:
    :param flipper_angle: [fl, fr, rl, rr]
    :return:
    '''

    if degree == True:
        flipper_angle = torch.deg2rad(flipper_angle)

    points = torch.zeros((4, 3))
    points[:, 0] = flipper_joint_x + flipper_length * torch.cos(flipper_angle)
    points[:, 1] = flipper_joint_y
    points[:, 2] = flipper_length * torch.sin(flipper_angle)

    return points * torch.tensor([
        [1, 1, 1],
        [1, -1, 1],
        [-1, 1, 1],
        [-1, -1, 1],
    ])
