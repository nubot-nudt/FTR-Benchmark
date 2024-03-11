import numpy as np

try:
    from omniisaacgymenvs.utils.terrain_utils.terrain_utils import convert_heightfield_to_trimesh, SubTerrain
    from omniisaacgymenvs.utils.terrain_utils import terrain_utils
    from omni.isaac.core.prims import XFormPrim
    from pxr import UsdPhysics, Sdf, Gf, PhysxSchema
    _has_isaac = True
except ImportError:
    _has_isaac = False
    pass

_terrain_num = 0


class Terrain():
    def __init__(self, num_terains, terrain_width, terrain_length, horizontal_scale, vertical_scale, slope_threshold=1.0):
        self.num_terains = num_terains
        self.terrain_width = terrain_width
        self.terrain_length = terrain_length
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.slope_threshold = slope_threshold

        self.num_rows = int(terrain_width / horizontal_scale)
        self.num_cols = int(terrain_length / horizontal_scale)
        self.heightfield = np.zeros((num_terains * self.num_rows, self.num_cols), dtype=np.int16)
    def new_sub_terrain(self):
        return SubTerrain(width=self.num_rows, length=self.num_cols, vertical_scale=self.vertical_scale,
                          horizontal_scale=self.horizontal_scale)
    def set_terrain(self, index, type_name, **kwargs):
        from . import register
        new_heightfield = getattr(register, f'{type_name}_terrain')(self.new_sub_terrain(), **kwargs).height_field_raw
        if index != 0:
            self.heightfield[index * self.num_rows:(index+1) * self.num_rows, :] = new_heightfield + self.heightfield[index * self.num_rows - 1].mean()
        else:
            self.heightfield[index * self.num_rows:(index + 1) * self.num_rows, :] = new_heightfield

    def add_terrain_to_stage(self, stage, position=np.array([0.0, 0, 0]), orientation=np.array([1, 0.0, 0.0, 0])):
        vertices, triangles = convert_heightfield_to_trimesh(self.heightfield, horizontal_scale=self.horizontal_scale,
                                                             vertical_scale=self.vertical_scale, slope_threshold=self.slope_threshold)
        num_faces = triangles.shape[0]
        global _terrain_num
        prim = f"/World/terrain{_terrain_num}"
        _terrain_num += 1
        terrain_mesh = stage.DefinePrim(prim, "Mesh")
        terrain_mesh.GetAttribute("points").Set(vertices)
        terrain_mesh.GetAttribute("faceVertexIndices").Set(triangles.flatten())
        terrain_mesh.GetAttribute("faceVertexCounts").Set(np.asarray([3] * num_faces))

        terrain = XFormPrim(prim_path=prim,
                            name="terrain",
                            position=position,
                            orientation=orientation)

        UsdPhysics.CollisionAPI.Apply(terrain.prim)
        # collision_api = UsdPhysics.MeshCollisionAPI.Apply(terrain.prim)
        # collision_api.CreateApproximationAttr().Set("meshSimplification")
        physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(terrain.prim)
        physx_collision_api.GetContactOffsetAttr().Set(0.002)
        physx_collision_api.GetRestOffsetAttr().Set(0.00)
        return prim
class TerrainInfo():
    def __init__(self, cfg):
        self.cfg = cfg
        self.position = cfg['position']
        self.orientation = cfg['orient']
        self.terrain = Terrain(len(cfg['terrain_list']), **cfg['base_config'])

        if _has_isaac == True:
            for i, t in enumerate(cfg['terrain_list']):
                self.terrain.set_terrain(i, **t)
    def add_terrain_to_stage(self, stage):
        return self.terrain.add_terrain_to_stage(stage, self.position, self.orientation)


# heightfield[0:num_rows, :] = random_uniform_terrain(new_sub_terrain(), min_height=-0.2, max_height=0.2, step=0.2,
#                                                     downsampled_scale=0.5).height_field_raw
# heightfield[num_rows:2 * num_rows, :] = sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
# heightfield[2 * num_rows:3 * num_rows, :] = pyramid_sloped_terrain(new_sub_terrain(), slope=-0.5).height_field_raw
# heightfield[3 * num_rows:4 * num_rows, :] = discrete_obstacles_terrain(new_sub_terrain(), max_height=0.5,
#                                                                        min_size=1., max_size=5.,
#                                                                        num_rects=20).height_field_raw
# heightfield[4 * num_rows:5 * num_rows, :] = wave_terrain(new_sub_terrain(), num_waves=2.,
#                                                          amplitude=1.).height_field_raw
# heightfield[5 * num_rows:6 * num_rows, :] = stairs_terrain(new_sub_terrain(), step_width=0.75,
#                                                            step_height=-0.5).height_field_raw
# heightfield[6 * num_rows:7 * num_rows, :] = pyramid_stairs_terrain(new_sub_terrain(), step_width=0.75,
#                                                                    step_height=-0.5).height_field_raw
# heightfield[7 * num_rows:8 * num_rows, :] = stepping_stones_terrain(new_sub_terrain(), stone_size=1.,
#                                                                     stone_distance=1., max_height=0.5,
#                                                                     platform_size=0.).height_field_raw