from omni.isaac.range_sensor import  _range_sensor

from scipy.spatial.transform import Rotation

from pumbaa.utils.prim import *

import numpy as np

lidarInterface = _range_sensor.acquire_lidar_sensor_interface()



class PointCloudHelper():

    t = 0.1

    compensation = np.array([1000, 1000])

    def __init__(self, lidars):
        '''
        only lidar with y-axis rotation of 90 is supported
        :param lidar: path of lidar that type is str
        '''
        self.lidars = lidars
        self.map = np.zeros(self.compensation * 2)
        self.num = np.zeros(self.compensation * 2)

    def get_map(self):
        return self.map / self.num
        # return self.map

    def compute_map(self):
        for path in self.lidars:
            coordinate, height = self.get_point_with_world_position(path)
            offset = self.compensation
            for index, i in enumerate(coordinate):
                c = (i + offset).astype(np.int32)

                # self.map[c[0], c[1]] = max( self.map[c[0], c[1]], height[index])

                self.map[c[0], c[1]] += height[index]
                self.num[c[0], c[1]] += 1

    def get_point_with_world_position(self, path):

        pointcloud = lidarInterface.get_point_cloud_data(path)
        pointcloud = pointcloud.reshape((-1, 3))

        r = Rotation.from_euler('xyz', get_rotate_value(path), degrees=True)
        tr = get_translate_value(path)

        pointcloud = r.apply(pointcloud, inverse=True)
        coordinate = np.floor((pointcloud[:, :2] + (tr[0], tr[1])) / self.t)
        height = tr[2] + pointcloud[:, 2]

        # coordinate = np.floor(pointcloud[:, -1:0:-1] / self.t)
        # coordinate = coordinate - np.array([tr[0], tr[1]])
        # height = tr[2] - pointcloud[:, 0]

        return coordinate, height