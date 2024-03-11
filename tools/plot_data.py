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

get_names = ['vel_roll', 'vel_pitch']
datas = {key:deque(maxlen=300) for key in get_names}


@reloading(every=10)
def get_preprocessing():

    preprocessing = [
        # lambda x: x**2,
        # lambda x: np.diff(x),
    ]
    return preprocessing

def update(frame):

    state = RobotState(get_keyboard_obs())

    for key in datas:
        datas[key].append(getattr(state, key))

    col = 3 if len(get_names) >= 3 else len(get_names)
    row = len(get_names) // col

    for i, key in enumerate(datas):
        ax = plt.subplot(row, col, i+1)
        ax.clear()
        data = np.array(datas[key])
        for f in get_preprocessing():
            data = f(data)
        plt.plot(data)
        plt.title(key)

ani = FuncAnimation(fig, update, frames=range(30), interval=100)
plt.show()