import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from utils.net.http_client import get_keyboard_obs

fig = plt.figure(figsize=(8, 4))
# ax = fig.add_subplot(111)

def update(frame):
    plt.cla()

    obs = get_keyboard_obs()

    array_2d = obs['img']

    gradient_x, gradient_y = np.gradient(array_2d)

    # 可视化原始数组和梯度
    ax = plt.subplot(131)
    plt.imshow(array_2d, cmap='gray'), plt.title('Original Array')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()

    ax = plt.subplot(132)
    plt.imshow(gradient_x, cmap='gray'), plt.title('Gradient in X direction')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_axis_off()

    ax = plt.subplot(133)
    plt.imshow(gradient_y, cmap='gray'), plt.title('Gradient in Y direction')

ani = FuncAnimation(fig, update, frames=range(30), interval=100)
plt.show()