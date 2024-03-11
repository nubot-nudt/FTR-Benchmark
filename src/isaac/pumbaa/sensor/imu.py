import numpy as np
from omni.isaac.core.scenes import Scene

class IMU():
    '''
    @bug 获取的数据全都为0
    '''
    def __init__(self, robot_prim_path, scene: Scene):
        from omni.isaac.sensor import IMUSensor
        self.imu_sensor = IMUSensor(
            prim_path=f"{robot_prim_path}/pumbaa_wheel/imu_link/imu_sensor",
            name="imu",
            frequency=60,
            translation=np.array([0, 0, 0]),
        )
        scene.add(self.imu_sensor)

    def initialize(self):
        self.imu_sensor.initialize()

    def get_lin_acc(self):
        return self.imu_sensor.get_current_frame()['lin_acc']

    def get_ang_vel(self):
        return self.imu_sensor.get_current_frame()['ang_vel']