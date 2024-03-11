
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5

from PyQt5.QtWidgets import QApplication

from gui.opview import OpView

if __name__ == '__main__':

    app = QApplication(sys.argv)

    main_view = OpView()
    main_view.show()

    sys.exit(app.exec())