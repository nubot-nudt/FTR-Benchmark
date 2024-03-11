import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from utils.net.http_client import get_keyboard_obs


if __name__ == '__main__':
    print(get_keyboard_obs()['flipper_velocities'])
