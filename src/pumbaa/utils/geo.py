import torch

def point_in_rotated_ellipse(x, y, h, k, a, b, theta):
    '''
    其中 (h, k) 是椭圆的中心坐标，a 和 b 分别是椭圆在旋转前 x 轴和 y 轴上的半长轴和半短轴的长度，theta 是椭圆的旋转角度（弧度制）。
    '''
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    term1 = ((x - h) * cos_theta + (y - k) * sin_theta) ** 2 / a ** 2
    term2 = ((x - h) * sin_theta - (y - k) * cos_theta) ** 2 / b ** 2
    return term1 + term2 <= 1
