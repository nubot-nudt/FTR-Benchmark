
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

import yaml

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5

from PyQt5.QtWidgets import QApplication
from gui.mapview import MapView

from utils.shell import execute_command

if __name__ == '__main__':

    app = QApplication(sys.argv)

    main_view = MapView()
    main_view.show()

    app.exec()

    from gui.mapview import is_launch_cmd, selected_file
    if is_launch_cmd == True:
        with open('data/gui.yaml', 'r') as f:
            python_path = yaml.safe_load(f)['python']

        execute_command(f'{python_path} scripts/teleop_play.py {selected_file}')