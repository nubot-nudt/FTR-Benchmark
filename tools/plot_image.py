import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from reloading import reloading

from autonomic.predictor.baselink import RobotState
from utils.net.http_client import get_keyboard_obs

fig = plt.figure()

def update(frame):

    state = RobotState(get_keyboard_obs())

    plt.imshow(state['camera_img'])

ani = FuncAnimation(fig, update, frames=range(30), interval=100)
plt.show()