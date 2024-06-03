

import os
import fire
from loguru import logger

from ptask_common.utils.tui.select_info import select_asset_path
from ptask_common.utils.common.asset import update_lastest_map_config


def main(config_path: str=None):
    if config_path is None:
        config_path = select_asset_path()

    asset_name = os.path.basename(config_path).removesuffix('.yaml')

    from omni.isaac.kit import SimulationApp
    simulation_app = SimulationApp()

    import omni
    from omni.isaac.core import World
    from ptask_envs.pumbaa.utils.prim import add_usd, update_collision
    from ptask_envs.pumbaa.utils.asset import AssetEntry
    from ptask_envs.pumbaa.utils.map import MapHelper

    world = World(stage_units_in_meters=1.0)
    scene = world.scene
    stage = omni.usd.get_context().get_stage()
    omni.kit.commands.execute('AddPhysicsSceneCommand', stage=stage, path='/World/PhysicsScene')
    asset = AssetEntry(file=config_path)

    if asset.is_add_ground_plane:
        scene.add_default_ground_plane()

    for name, usd in asset.obstacles.items():
        add_usd(usd)
    for name, terrain in asset.terrains.items():
        p = terrain.add_terrain_to_stage(stage)

    world.reset()
    for _ in range(10):
        world.step(render=True)

    map_helper = MapHelper(rebuild=True, **asset.map)

    for name, usd in asset.obstacles.items():
        update_collision(usd)
        # for path in find_matching_prim_paths(f'/World/{usd.name}/*/*/collisions'):
        #     attr = get_prim_at_path(path).GetAttribute('physics:approximation')
        #     attr.Set(attr.Get())

    for i in range(1, 3):
        world.step(render=True)
        logger.info(f'{i} step')
        map_helper.compute_map()

    map_helper.save()
    update_lastest_map_config(asset_name)

    print(asset.name)

    simulation_app.close()


if __name__ == '__main__':
    fire.Fire(main)
