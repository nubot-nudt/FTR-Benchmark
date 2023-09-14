
import numpy as np
from shapely.geometry import Point, Polygon

try:
    from omni.isaac.debug_draw import _debug_draw
except:
    pass

class Rectangle():
    def __init__(self, x_range, y_range):

        self.x_range = x_range
        self.y_range = y_range

        assert self.x_range.shape == 2 and self.y_range.shape == 2, \
            f'x_range.shape != 2 or y_range.shape != 2'

    def draw(self):
        drawer = _debug_draw.acquire_debug_draw_interface()

        a = [(self.x_range[0], self.y_range[0], 0.1),
             (self.x_range[0], self.y_range[1], 0.1),
             (self.x_range[1], self.y_range[1], 0.1),
             (self.x_range[1], self.y_range[0], 0.1)]
        drawer.draw_lines(a, a[1:]+[a[0]], [(1, 0, 0, 1)] * 4, [5] * 4)

    def contains(self, point):
        return point[0] in range(self.x_range[0], self.x_range[1]) \
            and point[1] in range(self.y_range[0], self.y_range[1])


class Geometryhelper():

    def __init__(self,
                 x_range=None, y_range=None,
                 target_point=None, target_x_range=None, target_y_range=None):

        self.x_range = np.array(x_range)
        self.y_range = np.array(y_range)
        self.target_point = np.array(target_point)
        self.target_x_range = np.array(target_x_range)
        self.target_y_range = np.array(target_y_range)

        if x_range is not None and y_range is not None:
            assert self.x_range.shape == self.y_range.shape,\
                f'x_range.shape={self.x_range.shape} != y_range.shape={self.y_range.shape}'

            self.range_polygon = Polygon([
                 (self.x_range[0], self.y_range[0]),
                 (self.x_range[0], self.y_range[1]),
                 (self.x_range[1], self.y_range[1]),
                 (self.x_range[1], self.y_range[0])
            ])
        else:
            raise RuntimeError('x_range and y_range must be not None')

        self.target_range_polygon = Polygon([
             (self.target_x_range[0], self.target_y_range[0]),
             (self.target_x_range[0], self.target_y_range[1]),
             (self.target_x_range[1], self.target_y_range[1]),
             (self.target_x_range[1], self.target_y_range[0])
        ])


    def draw(self):
        drawer = _debug_draw.acquire_debug_draw_interface()

        drawer.draw_points([(*self.target_point, 0.1)], [(0, 1, 0, 1)], [10])

        a = [(self.x_range[0], self.y_range[0], 0.1),
             (self.x_range[0], self.y_range[1], 0.1),
             (self.x_range[1], self.y_range[1], 0.1),
             (self.x_range[1], self.y_range[0], 0.1)]
        drawer.draw_lines(a, a[1:]+[a[0]], [(1, 0, 0, 1)] * 4, [5] * 4)

        a = [(self.target_x_range[0], self.target_y_range[0], 0.1),
             (self.target_x_range[0], self.target_y_range[1], 0.1),
             (self.target_x_range[1], self.target_y_range[1], 0.1),
             (self.target_x_range[1], self.target_y_range[0], 0.1)]
        drawer.draw_lines(a, a[1:] + [a[0]], [(0, 1, 0, 1)] * 4, [5] * 4)

    def is_in_range(self, point):
        return self.range_polygon.contains(Point(point))

    def is_in_target(self, point):
        return self.target_range_polygon.contains(Point(point))
