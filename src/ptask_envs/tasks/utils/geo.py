import torch

from ptask_common.utils.tensor import to_tensor


def distance_to_interval(x, a, b):
    return torch.max(0, torch.min(x - a, b - x))


def point_in_rotated_ellipse(x, y, h, k, a, b, theta):
    '''
    其中 (h, k) 是椭圆的中心坐标，a 和 b 分别是椭圆在旋转前 x 轴和 y 轴上的半长轴和半短轴的长度，theta 是椭圆的旋转角度（弧度制）。
    '''
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    term1 = ((x - h) * cos_theta + (y - k) * sin_theta) ** 2 / a ** 2
    term2 = ((x - h) * sin_theta - (y - k) * cos_theta) ** 2 / b ** 2
    return term1 + term2 <= 1


def distance_to_line_2d(A, B, C):
    """
    求C到AB的距离
    :param A:
    :param B:
    :param C:
    :return:
    """
    A = to_tensor(A)[:2]
    B = to_tensor(B)[:2]
    C = to_tensor(C)[:2]

    # 将点 A 和 B 转换为向量形式
    AB = B - A

    # 计算直线 AB 的长度的平方
    AB_length_squared = torch.sum(AB ** 2)

    # 如果直线 AB 的长度为 0，直接返回 C 到 A 点的距离
    if AB_length_squared == 0:
        return torch.norm(C - A)

    # 计算点 C 到 A 点的向量
    AC = C - A

    # 计算点 C 在直线 AB 上的投影点 D
    t = torch.sum(AC * AB) / AB_length_squared
    D = A + t * AB

    # 计算点 C 到直线 AB 的距离
    distance = torch.norm(C - D)

    return distance
