
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from PyQt5.QtWidgets import QApplication
from gui.captureview import CaptureView

if __name__ == '__main__':

    app = QApplication(sys.argv)

    main_view = CaptureView()
    main_view.show()

    sys.exit(app.exec())

