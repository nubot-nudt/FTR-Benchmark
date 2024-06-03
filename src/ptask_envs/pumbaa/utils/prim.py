import os
from itertools import chain

from loguru import logger
import pysnooper

from omni.isaac.core.utils.prims import create_prim, get_prim_at_path, find_matching_prim_paths
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.rotations import euler_angles_to_quat

from pxr import Usd

from .usd import UsdPrimInfo


def set_material_friction(path, v):
    '/World/v5_2_17/pumbaa_wheel/Looks/flipper_material.physics:dynamicFriction'
    prim = get_prim_at_path(path)
    prim.GetAttribute('physics:dynamicFriction').Set(v)
    prim.GetAttribute('physics:staticFriction').Set(v)


def set_joint_stiffness(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute('drive:angular:physics:stiffness').Set(v)

def set_joint_damping(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute('drive:angular:physics:damping').Set(v)

def set_joint_max_vel(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute('physxJoint:maxJointVelocity').Set(v)


def update_collision(usd: UsdPrimInfo, is_to_convex=False):
    collision_paths = [
        f'{usd.prim_path}/*/*/collisions',
        f'{usd.prim_path}/*/*/*/collisions',
        f'{usd.prim_path}/*/*/*/*/collisions',
        f'{usd.prim_path}/*/*/*/*/*/collisions',
    ]
    for path in chain(*[find_matching_prim_paths(i) for i in collision_paths]):
        # logger.debug(f'{path} is updating')
        prim = get_prim_at_path(path)

        if is_to_convex == False:
            attr = prim.GetAttribute('physics:approximation')
            value = attr.Get()
            attr.Set(value)
        else:
            # TODO collision bug on GPU

            prim.GetAttribute('physics:approximation').Set("convexDecomposition")
            # prim.GetAttribute('physxConvexDecompositionCollision:hullVertexLimit').Set(64)
            # prim.GetAttribute('physxConvexDecompositionCollision:maxConvexHulls').Set(128)


def get_rotate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute('xformOp:rotateXYZ').Get()


def get_translate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute('xformOp:translate').Get()


def add_usd(usd: UsdPrimInfo, prim_path=None):
    if prim_path is None:
        prim_path = usd.prim_path

    orientation = usd.orient if len(usd.orient) == 4 else euler_angles_to_quat(usd.orient, degrees=True)

    create_prim(prim_path=prim_path, prim_type="Xform",
                position=usd.position,
                orientation=orientation,
                scale=usd.scale)

    assert os.path.exists(usd.path) == True, f'{usd.path} file Not Found'
    return add_reference_to_stage(os.path.abspath(usd.path), prim_path)


def get_joint_pos_state(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute('state:angular:physics:position').Get()


def set_joint_pos_state(prim, pos):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    prim.GetAttribute('state:angular:physics:position').Set(pos)
