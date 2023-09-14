
from omni.isaac.sensor import IMUSensor
from omni.isaac.core.scenes import Scene
from omni.isaac.core.utils.prims import find_matching_prim_paths

class IMU():
    def __init__(self, robot_prim_path, scene: Scene):
        self.imu_sensor = scene.add(
            IMUSensor(
            prim_path=f"{robot_prim_path}/pumbaa_wheel/imu/imu_sensor",
            name="imu",
        ))

    def get_v_w(self):
        print(self.imu_sensor.get_current_frame())
        raise NotImplementedError()