# -*- coding: utf-8 -*-
"""
====================================
@File Name ：camera.py
@Time ： 2024/3/5 下午7:33
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import numpy as np
import omni.isaac.core.utils.numpy.rotations as rot_utils

from pumbaa.common.default import baselink_prim_path

class Camera():
    def __init__(self, robot_prim_path):
        from omni.isaac.sensor import Camera
        self.camera_sensor = Camera(
            prim_path=f"{robot_prim_path}/{baselink_prim_path}/camera",
            position=np.array([0.34, 0.0, 0.08]),
            frequency=20,
            resolution=(256, 256),
            orientation=rot_utils.euler_angles_to_quats(np.array([90, -90, 0]), degrees=True),
        )

    def initialize(self):
        self.camera_sensor.initialize()
        # self.camera_sensor.add_motion_vectors_to_frame()

    def get_image(self):
        return self.camera_sensor.get_rgba()[:, :, :3]