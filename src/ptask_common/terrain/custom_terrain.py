# -*- coding: utf-8 -*-
"""
====================================
@File Name ：custom_terrain.py
@Time ： 2024/4/8 下午4:16
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
from ptask_envs.omniisaacgymenvs.utils.terrain_utils import terrain_utils as tu
from .base_terrain import *


def train_mixed_terrain(subterrain, env_width=2, env_length=2, num_terrains=5, num_levels=5, border_size=2,
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
            step_height = 0.05 + 0.35 * difficulty

            terrain_gen = [
                lambda: flat_terrain(terrain, h1),
                lambda: batten_terrain(terrain, 0.4, h1),
                lambda: unilateral_step_terrain(terrain, h2, i % 2),
                lambda: plum_terrain(terrain, 0, step_height, 5),
                lambda: tu.random_uniform_terrain(terrain, min_height=-step_height / 4, max_height=step_height / 2,
                                                  step=0.05, downsampled_scale=0.4),
                lambda: tu.wave_terrain(terrain, num_waves=4, amplitude=step_height * 1 / 2),

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
