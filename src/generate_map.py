'''
    本程序用于生成离线地图，配置文件路经./assets/config，生成的地图文件存放在./assets/map中
'''

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
from utils.path import apply_project_directory
apply_project_directory()

if len(sys.argv) == 1:
    from tui.select_info import select_asset_path
    config_path = select_asset_path()

elif len(sys.argv) == 2:
    config_path = sys.argv[1]

else:
    sys.exit(1)


from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp()

import omni
from omni.isaac.core import World

from isaacsim_ext.prim import add_usd, update_collision
from pumbaa.common.asset import AssetEntry

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

if __name__ == '__main__':

    world.reset()
    # for _ in range(100):
    #     world.step(render=True)

    from pumbaa.helper.map import *

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

    print(asset.name)

    simulation_app.close()
