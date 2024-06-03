# -*- coding: utf-8 -*-
"""
====================================
@File Name ：curriculum_terrain.py
@Time ： 2024/3/23 下午2:11
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""

from ptask_envs.omniisaacgymenvs.utils.terrain_utils import terrain_utils as tu
from .base_terrain import *


def curriculum_mixed_terrain(subterrain, env_width=5, env_length=5, num_terrains=9, num_levels=7, border_size=1.8,
                             seed=40):
    np.random.seed(seed)

    border = int(border_size / subterrain.horizontal_scale)
    width_per_env_pixels = int(env_width / subterrain.horizontal_scale)
    length_per_env_pixels = int(env_length / subterrain.horizontal_scale)

    for j in range(num_terrains):
        for i in range(num_levels):
            terrain = SubTerrain(
                "terrain",
                width=width_per_env_pixels,
                length=length_per_env_pixels,
                vertical_scale=subterrain.vertical_scale,
                horizontal_scale=subterrain.horizontal_scale,
            )
            difficulty = i / num_levels

            slope = 0.1 + difficulty * 0.8
            step_height = 0.05 + 0.25 * difficulty
            discrete_obstacles_height = 0.1 + difficulty * 0.3

            if j == 0:
                tu.pyramid_sloped_terrain(terrain, slope=-slope, platform_size=1.8)
            elif j == 1:
                tu.pyramid_sloped_terrain(terrain, slope=+slope, platform_size=1.8)
            elif j == 2:
                tu.random_uniform_terrain(terrain, min_height=-0.1, max_height=0.1, step=0.03, downsampled_scale=0.2)
            elif j == 3:
                tu.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=-step_height, platform_size=2.5)
            elif j == 4:
                tu.pyramid_stairs_terrain(terrain, step_width=0.31, step_height=step_height, platform_size=2.5)
            elif j == 5:
                tu.discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1.0, 2.0, 10, platform_size=1.8)
            elif j == 6:
                flat_terrain(terrain, discrete_obstacles_height)
            elif j == 7:
                batten_terrain(terrain, 0.4, discrete_obstacles_height)
            elif j == 8:
                tu.discrete_obstacles_terrain(terrain, discrete_obstacles_height, 1.0, 2.0, 20, platform_size=1.8)

            # Heightfield coordinate system
            start_x = border + border * i + i * length_per_env_pixels
            end_x = border + border * i + (i + 1) * length_per_env_pixels
            start_y = border + j * width_per_env_pixels
            end_y = border + (j + 1) * width_per_env_pixels
            subterrain.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    np.random.seed(None)

    return subterrain


def curriculum_rugged_terrain(subterrain, env_width=3, env_length=3, num_terrains=7, num_levels=7, border_size=1.8,
                              seed=40):
    np.random.seed(seed)

    border = int(border_size / subterrain.horizontal_scale)
    width_per_env_pixels = int(env_width / subterrain.horizontal_scale)
    length_per_env_pixels = int(env_length / subterrain.horizontal_scale)

    for j in range(num_terrains):
        for i in range(num_levels):
            terrain = SubTerrain(
                "terrain",
                width=width_per_env_pixels,
                length=length_per_env_pixels,
                vertical_scale=subterrain.vertical_scale,
                horizontal_scale=subterrain.horizontal_scale,
            )
            difficulty = i / num_levels

            slope = 0.1 + difficulty * 0.8
            step_height = 0.05 + 0.35 * difficulty
            discrete_obstacles_height = 0.1 + difficulty * 0.3

            terrain_gen = [
                lambda: plum_terrain(terrain, step_height, 5),
                lambda: tu.random_uniform_terrain(terrain, min_height=-step_height / 2.5, max_height=step_height / 3,
                                                  step=0.05, downsampled_scale=0.4),
                lambda: tu.wave_terrain(terrain, num_waves=2, amplitude=step_height * 3 / 5),
                lambda: plum_terrain(terrain, step_height, 8),
                lambda: tu.random_uniform_terrain(terrain, min_height=-step_height / 3, max_height=step_height / 3,
                                                  step=0.03, downsampled_scale=0.25),
                lambda: tu.wave_terrain(terrain, num_waves=4, amplitude=step_height * 1 / 3),
                lambda: plum_terrain(terrain, step_height, 10),
            ]

            terrain_gen[j % len(terrain_gen)]()

            # Heightfield coordinate system
            start_x = border + border * i + i * length_per_env_pixels
            end_x = border + border * i + (i + 1) * length_per_env_pixels
            start_y = border + border * j + j * width_per_env_pixels
            end_y = border + border * j + (j + 1) * width_per_env_pixels
            subterrain.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    np.random.seed(None)

    return subterrain


def curriculum_stairs_terrain(subterrain, env_width=5, env_length=5, num_terrains=7, num_levels=7, border_size=1.8,
                              seed=40):
    np.random.seed(seed)

    border = int(border_size / subterrain.horizontal_scale)
    width_per_env_pixels = int(env_width / subterrain.horizontal_scale)
    length_per_env_pixels = int(env_length / subterrain.horizontal_scale)

    step_widths = [(k + 1) / num_terrains * 0.4 for k in range(num_terrains)]
    stair_angle = [(k + 1) / num_levels * (45 - 10) + 10 for k in range(num_levels)]

    for j in range(num_terrains):
        for i in range(num_levels):
            terrain = SubTerrain(
                "terrain",
                width=width_per_env_pixels,
                length=length_per_env_pixels,
                vertical_scale=subterrain.vertical_scale,
                horizontal_scale=subterrain.horizontal_scale,
            )

            step_width = step_widths[j]
            step_height = np.tan(stair_angle[i] / 180 * np.pi) * step_width

            tu.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height * (j % 2 - 0.5) * 2,
                                      platform_size=2.8)

            # Heightfield coordinate system
            start_x = border + border * i + i * length_per_env_pixels
            end_x = border + border * i + (i + 1) * length_per_env_pixels
            start_y = border + border * j + j * width_per_env_pixels
            end_y = border + border * j + (j + 1) * width_per_env_pixels
            subterrain.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    np.random.seed(None)

    return subterrain


def curriculum_steps_terrain(subterrain, env_width=3, env_length=3, num_terrains=7, num_levels=7, border_size=1.8,
                             seed=40):
    np.random.seed(seed)

    border = int(border_size / subterrain.horizontal_scale)
    width_per_env_pixels = int(env_width / subterrain.horizontal_scale)
    length_per_env_pixels = int(env_length / subterrain.horizontal_scale)

    for j in range(num_terrains):
        for i in range(num_levels):
            terrain = SubTerrain(
                "terrain",
                width=width_per_env_pixels,
                length=length_per_env_pixels,
                vertical_scale=subterrain.vertical_scale,
                horizontal_scale=subterrain.horizontal_scale,
            )
            difficulty = (i + 1) / num_levels
            h1 = 0.1 + difficulty * 0.3
            h2 = 0.1 + difficulty * 0.2

            terrain_gen = [
                lambda: flat_terrain(terrain, h1),
                lambda: flat_terrain(terrain, -h1),
                lambda: tu.discrete_obstacles_terrain(terrain, h2 / 2, 1.2, 2.0, 8,
                                                      platform_size=1.2),
                lambda: batten_terrain(terrain, 0.4, h1),
                lambda: batten_terrain(terrain, 1.0, -h1),
                lambda: unilateral_step_terrain(terrain, h2, i % 2),
                lambda: unilateral_step_terrain(terrain, -h2, i % 2),
            ]

            terrain_gen[j % len(terrain_gen)]()

            # Heightfield coordinate system
            start_x = border + border * i + i * length_per_env_pixels
            end_x = border + border * i + (i + 1) * length_per_env_pixels
            start_y = border + border * j + j * width_per_env_pixels
            end_y = border + border * j + (j + 1) * width_per_env_pixels
            subterrain.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    np.random.seed(None)

    return subterrain


def curriculum_terrain_generate(subterrain,
                                terrain_gen_func,
                                env_width=3,
                                env_length=3,
                                num_terrains=7,
                                num_levels=7,
                                border_size=1.8,
                                seed=40):
    np.random.seed(seed)

    border = int(border_size / subterrain.horizontal_scale)
    width_per_env_pixels = int(env_width / subterrain.horizontal_scale)
    length_per_env_pixels = int(env_length / subterrain.horizontal_scale)

    for j in range(num_terrains):
        for i in range(num_levels):
            terrain = SubTerrain(
                "terrain",
                width=width_per_env_pixels,
                length=length_per_env_pixels,
                vertical_scale=subterrain.vertical_scale,
                horizontal_scale=subterrain.horizontal_scale,
            )
            difficulty = (i + 1) / num_levels
            terrain_gen_func(terrain, difficulty=difficulty, terrain_index=j, level_index=i)

            # Heightfield coordinate system
            start_x = border + border * i + i * length_per_env_pixels
            end_x = border + border * i + (i + 1) * length_per_env_pixels
            start_y = border + border * j + j * width_per_env_pixels
            end_y = border + border * j + (j + 1) * width_per_env_pixels
            subterrain.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

    np.random.seed(None)

    return subterrain


def curriculum_steps_up_terrain(subterrain, min_h=0.1, max_h=0.4, **kwargs):
    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        h = min_h + difficulty * (max_h - min_h)
        return flat_terrain(terrain, h)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_steps_down_terrain(subterrain, min_h=0.1, max_h=0.4, **kwargs):
    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        h = min_h + difficulty * (max_h - min_h)
        return flat_terrain(terrain, -h)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_unilateral_steps_terrain(subterrain, min_h=0.1, max_h=0.4, length=-1, **kwargs):
    num_levels = kwargs['num_levels']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        h = min_h + difficulty * (max_h - min_h)
        return unilateral_step_terrain(terrain, h, (level_index + 1) % 2, length)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_stairs_up_terrain(subterrain, min_step_width=0.2, max_step_width=0.3, min_angle=20,
                                 max_angle=30, **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        step_width = np.linspace(min_step_width, max_step_width, num_terrains)[terrain_index]
        step_height = step_width * np.tan(np.linspace(min_angle, max_angle, num_terrains) / 180 * np.pi)[level_index]

        # return tu.pyramid_stairs_terrain(terrain, step_width=step_width, step_height=step_height[level_index],
        #                                  platform_size=platform_size)
        return stairs3_terrain(terrain, step_width, step_height, num_step=5)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_stairs_down_terrain(subterrain, min_step_width=0.2, max_step_width=0.3, min_angle=20,
                                 max_angle=30, **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        step_width = np.linspace(min_step_width, max_step_width, num_terrains)[terrain_index]
        step_height = step_width * np.tan(np.linspace(min_angle, max_angle, num_terrains) / 180 * np.pi)[level_index]
        return stairs3_terrain(terrain, step_width, -step_height, num_step=5)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_uplift_terrain(subterrain, min_width=0.2, max_width=0.3, min_height=0.2, max_height=0.4, offset=0.2,
                              **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        step_width = np.linspace(min_width, max_width, num_terrains)[terrain_index]
        step_height = np.linspace(min_height, max_height, num_levels)[level_index]

        return uplift_terrain(terrain, width=step_width, height=step_height, offset=offset)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_plum_piles_terrain(subterrain, min_height=0.2, max_height=0.4, **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']
    seed = kwargs['seed']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        mh = np.linspace(min_height, max_height, num_levels)[level_index]
        return plum_terrain(terrain, min_height=0, max_height=mh, num=terrain_index + 4,
                            seed=seed + terrain_index * num_terrains + level_index)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_waves_terrain(subterrain, min_amplitude=0.05, max_amplitude=0.2, **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']
    seed = kwargs['seed']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        amplitude = np.linspace(min_amplitude, max_amplitude, num_levels)[level_index]
        return tu.wave_terrain(terrain, num_waves=terrain_index + 1, amplitude=amplitude)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)


def curriculum_rails_terrain(subterrain, min_angle=0.05, max_angle=0.2, min_h=0.1, max_h=0.3, **kwargs):
    num_levels = kwargs['num_levels']
    num_terrains = kwargs['num_terrains']
    seed = kwargs['seed']

    def _terrain_gen_func(terrain, difficulty, terrain_index, level_index):
        angle = np.linspace(min_angle, max_angle, num_levels)[terrain_index]
        h = np.linspace(min_h, max_h, num_terrains)[level_index]
        return rail_terrain(terrain, angle=angle, height=h, width=0.1, direct=terrain_index % 2)

    return curriculum_terrain_generate(subterrain, _terrain_gen_func, **kwargs)
