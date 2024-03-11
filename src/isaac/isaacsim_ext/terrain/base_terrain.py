from omniisaacgymenvs.utils.terrain_utils.terrain_utils import SubTerrain
from random import Random


def flat_terrain(terrain: SubTerrain, height=0):
    terrain.height_field_raw[:, :] = height / terrain.vertical_scale
    return terrain


def batten_terrain(terrain: SubTerrain, width=0.2, height=0.2):
    center_x = terrain.width // 2
    center_y = terrain.length // 2

    offset = int(width / terrain.horizontal_scale // 2)
    terrain.height_field_raw[center_x - offset:center_x + offset, :] = height / terrain.vertical_scale
    return terrain


def unilateral_step_terrain(terrain: SubTerrain, height=0.2, side=0):
    center_y = terrain.length // 2
    if side == 0:
        terrain.height_field_raw[1:-1, :center_y] = height / terrain.vertical_scale
    else:
        terrain.height_field_raw[1:-1, center_y:] = height / terrain.vertical_scale
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
