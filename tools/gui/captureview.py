import pickle

import requests
import numpy as np

from loguru import logger

from PyQt5.QtWidgets import QWidget, QTableWidgetItem, QFileDialog
from PyQt5.QtCore import QTimer
from utils.tensor import to_list

from .ui.capture import Ui_Form


base_url = 'http://127.0.0.1:12345'

class CaptureView(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('capture')

        self.init_pose()
        self.init_map()

    def init_map(self):
        self.map_capture_state = 'stop'
        self.update_map_btn_state()

        self.ui.map_start.clicked.connect(self.on_click_map_start)
        self.ui.map_pause.clicked.connect(self.on_click_map_pause)
        self.ui.map_stop.clicked.connect(self.on_click_map_stop)
        self.ui.map_export.clicked.connect(self.on_click_map_export)

        self.map_capture_data_list = []

    def on_click_map_export(self):
        filename = QFileDialog.getSaveFileName(self, directory='data/map')
        with open(filename[0], 'wb') as f:
            pickle.dump(self.map_capture_data_list, f)

    @logger.catch
    def map_capture(self):
        if self.map_capture_state != 'running':
            return

        content = requests.get(f'{base_url}/obs').content
        data = pickle.loads(content)

        map_data = data['img']
        # print(map_data.shape)
        self.map_capture_data_list.append(map_data)

        self.ui.map_current_num.setText(str(len(self.map_capture_data_list)))

    def on_click_map_stop(self):

        if self.map_capture_state == 'running':
            self.map_capture_state = 'stop'
        elif self.map_capture_state == 'pause':
            self.map_capture_state = 'stop'
        else:
            raise NotImplementedError()

        self.map_capture_timer.stop()

        self.update_map_btn_state()

    def on_click_map_pause(self):

        if self.map_capture_state == 'running':
            self.map_capture_state = 'pause'
        else:
            raise NotImplementedError()

        self.update_map_btn_state()

    def on_click_map_start(self):

        if self.map_capture_state == 'stop':
            self.map_capture_state = 'running'

            t = self.ui.map_hz.value()
            self.map_capture_data_list.clear()

            self.map_capture_timer = QTimer()
            self.map_capture_timer.timeout.connect(self.map_capture)
            self.map_capture_timer.start(int(1000 / t))

        elif self.map_capture_state == 'pause':
            self.map_capture_state = 'running'
        else:
            raise NotImplementedError()

        self.update_map_btn_state()

    def update_map_btn_state(self):
        if self.map_capture_state == 'stop':
            self.ui.map_start.setEnabled(True)
            self.ui.map_pause.setDisabled(True)
            self.ui.map_stop.setDisabled(True)
        elif self.map_capture_state == 'running':
            self.ui.map_start.setDisabled(True)
            self.ui.map_pause.setEnabled(True)
            self.ui.map_stop.setEnabled(True)
        elif self.map_capture_state == 'pause':
            self.ui.map_start.setEnabled(True)
            self.ui.map_pause.setDisabled(True)
            self.ui.map_stop.setEnabled(True)
        else:
            raise NotImplementedError()


    def init_pose(self):
        self.pose_data = list()
        self.pose_data = [
            {
                'start_point': [0, 0, 0],
                'start_orient': [0, 0, 0],
                'target_point': [0, 0, 0],
                'target_orient': [0, 0, 0],
            },
        ]
        self.pose_data.clear()
        self.update_pose_table()
        self.init_pose_btn()

    def init_pose_btn(self):
        self.ui.fetch_btn.setText('起点')
        self.status = 0
        self.current_data = dict()
        self.ui.fetch_btn.clicked.connect(self.record)

        self.ui.export_btn.clicked.connect(self.export)

        self.ui.clear_btn.clicked.connect(self.clear)

        self.ui.delete_btn.clicked.connect(self.on_delete)

        self.ui.load_btn.clicked.connect(self.load)

    def load(self):
        filename = QFileDialog.getOpenFileName(self, directory='data/start_target')
        with open(filename[0], 'rb') as f:
            self.pose_data = pickle.load(f)
        self.update_pose_table()

    def on_delete(self):
        indexes = self.ui.start_and_target_tab.selectedIndexes()
        row_indexes = list(set(map(lambda x: x.row(), indexes)))
        row_indexes.sort(reverse=True)
        for index in row_indexes:
            self.ui.start_and_target_tab.removeRow(index)

    def clear(self):
        self.ui.start_and_target_tab.clearContents()
        self.pose_data.clear()
        self.update_pose_table()

    def export(self):
        filename = QFileDialog.getSaveFileName(self, directory='data/start_target')
        with open(filename[0], 'wb') as f:
            pickle.dump(self.pose_data, f)

    @logger.catch(default=(None, None))
    def get_current_pose(self):

        content = requests.get(f'{base_url}/obs').content
        data = pickle.loads(content)

        return np.round(data["pos"], 2), np.round(data["orient"], 2)

    def record(self):
        if self.status == 0:
            point, orient = self.get_current_pose()
            self.current_data['start_point'] = to_list(point)
            self.current_data['start_orient'] = to_list(orient)

            if point is None:
                return

            self.ui.fetch_btn.setText("终点")
            self.status = 1
        else:
            point, orient = self.get_current_pose()
            self.current_data['target_point'] = to_list(point)
            self.current_data['target_orient'] = to_list(orient)

            if point is None:
                return

            print(self.current_data)
            self.pose_data.append(self.current_data.copy())
            self.current_data.clear()

            self.update_pose_table()
            self.ui.fetch_btn.setText("起点")
            self.status = 0

    def update_pose_table(self):
        self.ui.start_and_target_tab.setRowCount(max(len(self.pose_data), 1))
        for index, item in enumerate(self.pose_data):
            self.ui.start_and_target_tab.setItem(index, 0, QTableWidgetItem(str(item['start_point'])))
            self.ui.start_and_target_tab.setItem(index, 1, QTableWidgetItem(str(item['start_orient'])))
            self.ui.start_and_target_tab.setItem(index, 2, QTableWidgetItem(str(item['target_point'])))
            self.ui.start_and_target_tab.setItem(index, 3, QTableWidgetItem(str(item['target_orient'])))
