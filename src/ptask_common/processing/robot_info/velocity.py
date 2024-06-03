import torch


def world_velocity_to_v_w(vels):
    vels = vels.view(-1, 6)
    return torch.stack([vels[:, 0:3].norm(dim=1), vels[:, 5]], dim=1)

def world_velocity_to_v(vels):
    vels = vels.view(-1, 6)
    v = vels[:, 0:3].norm(dim=1)

    if len(vels.shape) == 1:
        return v.view(-1)
    return v

def world_velocity_to_w(vels):
    vels = vels.view(-1, 6)
    w = vels[:, 5]

    if len(vels.shape) == 1:
        return w.view(-1)
    return w