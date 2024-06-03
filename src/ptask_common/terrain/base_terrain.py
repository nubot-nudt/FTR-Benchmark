from ptask_envs.omniisaacgymenvs.utils.terrain_utils.terrain_utils import SubTerrain
from random import Random
import numpy as np
from itertools import product


def rail_terrain(terrain, angle, height, width, direct=0):
    ta = np.tan(angle / 180 * np.pi)
    start_x = 0.5 * terrain.width * (1 - ta) * terrain.horizontal_scale
    w = int(width // terrain.horizontal_scale)

    terrain.height_field_raw[:, :w] = int(height // terrain.vertical_scale)
    terrain.height_field_raw[:, -w:] = int(height // terrain.vertical_scale)

    for i, j, in product(range(terrain.width), range(terrain.length)):
        x = i * terrain.horizontal_scale
        y = j * terrain.horizontal_scale
        d = np.abs(ta * y - x + start_x) / np.sqrt(1 + ta ** 2)
        if d < width:
            if direct == 0:
                terrain.height_field_raw[i, j] = int(height // terrain.vertical_scale)
            else:
                terrain.height_field_raw[i, -j] = int(height // terrain.vertical_scale)

    return terrain

def rail2_terrain(terrain, angle, height, width, direct=0):
    ta = np.tan(angle / 180 * np.pi)
    start_x = 0.5 * terrain.width * (1 - ta) * terrain.horizontal_scale
    w = int(width // terrain.horizontal_scale)

    terrain.height_field_raw[:, :w] = int(height // terrain.vertical_scale)
    terrain.height_field_raw[:, -w:] = int(height // terrain.vertical_scale)

    for i, j, in product(range(terrain.width), range(terrain.length)):
        x = i * terrain.horizontal_scale
        y = j * terrain.horizontal_scale
        d = np.abs(ta * y - x + start_x) / np.sqrt(1 + ta ** 2)
        if d < width / 2:
            if direct == 0:
                terrain.height_field_raw[i, j] = int(height // terrain.vertical_scale)
            else:
                terrain.height_field_raw[i, -j] = int(height // terrain.vertical_scale)

        if i == 0 or j == 0:
            terrain.height_field_raw[i, j] = 0

    return terrain


def plum_terrain(terrain, min_height=0, max_height=0.4, num=10, seed=40):
    random = Random(seed)

    e_width = terrain.width // num
    e_length = terrain.length // num

    for i, j in product(range(num), range(num)):
        w = e_width * i
        l = e_length * j
        h = (max_height - min_height) * random.random() + min_height
        h = h * ((i + 1) / num * 0.6 + 0.4)
        terrain.height_field_raw[w:w + e_width, l:l + e_length] = h / terrain.vertical_scale

        # print(terrain.height_field_raw[w, l])
    # print(terrain.height_field_raw)

    # terrain.height_field_raw = gaussian_filter(terrain.height_field_raw, 1)
    return terrain


def stairs2_terrain(terrain, step_width, step_height, max_width=100):
    """

    :param terrain:
    :param step_width: 楼梯每个台阶的宽度
    :param step_height: 楼梯每个台阶的高度
    :param width: 楼梯最大宽度
    :return:
    """
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = terrain.width // step_width
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height

        if i * step_width * terrain.horizontal_scale <= max_width:
            height += step_height

    terrain.height_field_raw[num_steps * step_width:, :] += height

    return terrain


def stairs3_terrain(terrain, step_width, step_height, num_step=5):
    step_width = int(step_width / terrain.horizontal_scale)
    step_height = int(step_height / terrain.vertical_scale)

    num_steps = min(terrain.width // step_width, num_step - 1)
    height = step_height
    for i in range(num_steps):
        terrain.height_field_raw[i * step_width: (i + 1) * step_width, :] += height
        height += step_height

    terrain.height_field_raw[num_steps * step_width:, :] += height

    return terrain


def flat_terrain(terrain: SubTerrain, height=0):
    terrain.height_field_raw[:, :] = height / terrain.vertical_scale
    return terrain


def batten_terrain(terrain: SubTerrain, width=0.2, height=0.2):
    center_x = terrain.width // 2
    center_y = terrain.length // 2

    offset = int(width / terrain.horizontal_scale // 2)
    terrain.height_field_raw[center_x - offset:center_x + offset, :] = height / terrain.vertical_scale
    return terrain


def uplift_terrain(terrain: SubTerrain, width=0.2, height=0.2, offset=0.1):
    start_x = int(offset // terrain.horizontal_scale)
    end_x = start_x + int(width // terrain.horizontal_scale)
    terrain.height_field_raw[start_x:end_x, :] = height / terrain.vertical_scale
    return terrain


def unilateral_step_terrain(terrain: SubTerrain, height=0.2, side=0, length=-1):
    center_y = terrain.length // 2
    length = int(length // terrain.horizontal_scale) if length > 0 else length
    if side == 0:
        terrain.height_field_raw[1:length, :center_y] = height / terrain.vertical_scale
    else:
        terrain.height_field_raw[1:length, center_y:] = height / terrain.vertical_scale
    return terrain


def random_steps_terrain(terrain: SubTerrain, min_width, max_width, min_length, max_length, min_height, max_height,
                         step_num, seed):
    random = Random(seed)

    min_x = min_width // terrain.horizontal_scale
    max_x = max_width // terrain.horizontal_scale
    min_y = min_length // terrain.horizontal_scale
    max_y = max_length // terrain.horizontal_scale
    min_z = min_height // terrain.vertical_scale
    max_z = max_height // terrain.vertical_scale

    for _ in range(step_num):
        x = random.randint(0, terrain.width - max_x)
        y = random.randint(0, terrain.length - max_y)
        w = random.randint(min_x, max_x)
        l = random.randint(min_y, max_y)
        h = random.randint(min_z, max_z)
        terrain.height_field_raw[x:x + w, y:y + l] = h

    return terrain
