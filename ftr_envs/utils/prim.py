import os
from itertools import chain

from omni.isaac.core.utils.prims import (
    create_prim,
    find_matching_prim_paths,
    get_prim_at_path,
)
from omni.isaac.core.utils.rotations import euler_angles_to_quat
from omni.isaac.core.utils.stage import add_reference_to_stage
from pxr import Usd


def set_material_color(path, color):
    prim = get_prim_at_path(f"{path}/Shader")
    prim.GetAttribute("inputs:diffuse_tint").Set(color)


def get_prim_radius(path):
    prim = get_prim_at_path(path)
    return prim.GetAttribute("radius").Get()


def set_render_radius(path, radius):
    prim = get_prim_at_path(path)
    prim.GetAttribute("radius").Set(radius)


def set_prim_invisible(path):
    prim = get_prim_at_path(path)
    prim.GetAttribute("visibility").Set("invisible")


def set_material_friction(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("physics:dynamicFriction").Set(v)
    prim.GetAttribute("physics:staticFriction").Set(v)


def set_joint_stiffness(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("drive:angular:physics:stiffness").Set(v)


def set_joint_damping(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("drive:angular:physics:damping").Set(v)


def set_joint_max_vel(path, v):
    prim = get_prim_at_path(path)
    prim.GetAttribute("physxJoint:maxJointVelocity").Set(v)


def get_rotate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("xformOp:rotateXYZ").Get()


def get_translate_value(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("xformOp:translate").Get()


def get_joint_pos_state(prim):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    return prim.GetAttribute("state:angular:physics:position").Get()


def set_joint_pos_state(prim, pos):
    if not isinstance(prim, Usd.Prim):
        prim = get_prim_at_path(prim)
    prim.GetAttribute("state:angular:physics:position").Set(pos)


def add_usd(usd_file, prim_path, pos=(0, 0, 0), orient=(1, 0, 0, 0), scale=(1, 1, 1)):
    orientation = orient if len(orient) == 4 else euler_angles_to_quat(orient, degrees=True)

    create_prim(
        prim_path=prim_path,
        prim_type="Xform",
        position=pos,
        orientation=orientation,
        scale=scale,
    )

    assert os.path.exists(usd_file) == True, f"{usd_file} file Not Found"
    return add_reference_to_stage(os.path.abspath(usd_file), prim_path)
