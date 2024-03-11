import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from autonomic.predictor.baselink import LSFPredictor, RobotState
from pumbaa.common.default import robot_width, robot_length
from processing.robot_info.robot_point import robot_flipper_positions, robot_to_world
from utils.net.http_client import get_keyboard_obs

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def update(frame):
    ax.clear()

    predictor = LSFPredictor()
    state = RobotState(get_keyboard_obs())

    points = predictor.get_all_points(state.height_map)

    robot_points = points[
        (points[:, 0] >= -robot_length / 2) & (points[:, 0] <= robot_length / 2) &
        (points[:, 0] >= -robot_width / 2) & (points[:, 0] <= robot_width / 2)
    ]

    a, b, c, d = predictor.calc_plane(robot_points)
    xx, yy = np.meshgrid(np.linspace(-robot_length/2, robot_length/2, 10), np.linspace(-robot_width/2, robot_width/2, 10))
    zz = (-a * xx - b * yy - d) / c

    plot_robot(state)

    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.plot_surface(xx, yy, zz, alpha=0.5, color='r')

    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_axis_off()


def plot_robot(state):
    flipper_points = robot_flipper_positions(state.flippers, degree=True)
    flipper_root_points = robot_flipper_positions(state.flippers, degree=True, flipper_length=0)
    flipper_points = robot_to_world(state.roll, state.pitch, flipper_points)
    flipper_root_points = robot_to_world(state.roll, state.pitch, flipper_root_points)

    for fp, frp in zip(flipper_points, flipper_root_points):
        ax.plot([fp[0], frp[0]], [fp[1], frp[1]], [fp[2], frp[2]], color='r')



ani = FuncAnimation(fig, update, frames=range(30), interval=100)
plt.show()