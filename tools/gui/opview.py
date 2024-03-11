import os, pickle, json
import requests

import matplotlib.pyplot as plt
import torch
from scipy.spatial import ConvexHull

from PyQt5.QtWidgets import QWidget, QGraphicsScene
from PyQt5.QtCore import QTimer

from .ui.opview import Ui_Form

base_url = 'http://127.0.0.1:12345'

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg

class OpView(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('op')

        self.init_btn()
        self.init_graphics_view()
        self.init_checkbtn()
        self.init_radio_btn()

        self.request_timer = QTimer(self)
        self.request_timer.timeout.connect(self.request_data)
        self.request_timer.start(100)

    def init_radio_btn(self):
        def on_click(radio):
            pass

        self.ui.plt_radio_map.clicked.connect(lambda : on_click(self.ui.plt_radio_map))
        self.ui.plt_radio_contact.clicked.connect(lambda : on_click(self.ui.plt_radio_contact))
    def init_checkbtn(self):
        def ai_flipper_on_change(is_check):
            try:
                requests.post(f'{base_url}/set_ai_flipper', data=pickle.dumps(is_check))
            except:
                self.ui.textBrowser.setText('服务器异常')

        self.ui.ai_flipper.toggled.connect(ai_flipper_on_change)
        ai_flipper_on_change(self.ui.ai_flipper.isChecked())

    def init_btn(self):
        btns = [i for i in dir(self.ui) if i.startswith('btn')]
        # print(btns)

        def on_click():

            if not hasattr(self, 'flipper'):
                return

            v =  self.ui.set_vel.value()
            angle = self.ui.set_angle.value()
            name: str = self.sender().objectName()[4:]

            try:

                if name == 'reset':
                    requests.get(f'{base_url}/reset')
                    self.last_click_v = 0
                    self.last_click_w = 0
                    self.last_click_flipper = torch.zeros((4,))
                    return

                action_vel_map = {
                    'stop': [0, 0],
                    'forward': [v, 0],
                    'backward': [-v, 0],
                    'left': [0, 2 * v],
                    'right': [0, -2 * v],
                }

                action_flipper_map = {
                    'fl_up': [angle, 0, 0, 0],
                    'fl_down': [-angle, 0, 0, 0],
                    'fr_up': [0, angle, 0, 0],
                    'fr_down': [0, -angle, 0, 0],
                    'rl_up': [0, 0, angle, 0],
                    'rl_down': [0, 0, -angle, 0],
                    'rr_up': [0, 0, 0, angle],
                    'rr_down': [0, 0, 0, -angle],
                }

                action_flipper_set_map = {
                    'flipper_reset': [0, 0, 0, 0],
                    'flipper_all_up': [60, 60, 60, 60],
                }

                if name in action_vel_map:
                    action = action_vel_map[name]
                    cmd = {
                        'vel_type': 'std',
                        'vels': torch.tensor(action),
                        'flipper_type': 'dt',
                        'flippers': torch.tensor([0, 0, 0, 0])
                    }

                elif name in action_flipper_map:
                    action = action_flipper_map[name]
                    cmd = {
                        'flipper_type': 'pos',
                        'flippers': torch.tensor(action) + self.flipper
                    }
                elif name in action_flipper_set_map:
                    action = action_flipper_set_map[name]
                    cmd = {
                        'flipper_type': 'pos',
                        'flippers': torch.tensor(action)
                    }
                else:
                    return

                requests.post(f'{base_url}/set_action', data=pickle.dumps(cmd))

            except requests.exceptions.RequestException:
                self.ui.textBrowser.setText('网络异常')
                return


        for btn in btns:
            getattr(self.ui, btn).clicked.connect(on_click)

    def print_(self, text):
        self.out_text = self.out_text + '\n' + text

    def request_data(self):

        try:
            content = requests.get(f'{base_url}/obs').content
        except requests.exceptions.RequestException:
            self.ui.textBrowser.clear()
            self.ui.textBrowser.setText('网络异常')
            return

        try:
            data = pickle.loads(content)
        except EOFError:
            return

        self.img = data['img']
        self.pos = data["pos"]
        self.orient = data["orient"]
        self.v = data["v"]
        self.w = data["w"]
        self.flipper = data["flipper"]

        if data.get('contacts', None) is not None:
            self.contacts = data["contacts"]

        self.update_view()

    def update_view(self):
        plt.cla()

        if self.ui.plt_radio_map.isChecked():
            plt.imshow(self.img)
        elif self.ui.plt_radio_contact.isChecked():
            self.show_contact_info()

        self.canvas.draw()

        for i, s in enumerate(['x', 'y', 'z']):
            getattr(self.ui, f'pos_{s}').setText(f'{self.pos[i]:-2.2f}')

        for i, s in enumerate(['x', 'y', 'z']):
            getattr(self.ui, f'orient_{s}').setText(f'{self.orient[i]:-2.2f}')

        self.ui.vel_v.setText(f'{self.v:-2.2f}')
        self.ui.vel_w.setText(f'{self.w:-2.2f}')

        for i, s in enumerate(['fl', 'fr', 'rl', 'rr']):
            getattr(self.ui, f'angle_{s}').setText(f'{self.flipper[i]:-2.2f}')

    def show_contact_info(self):
        robot_points = [
            [-0.38, -0.24], [-0.38, 0.24], [0.38, 0.24], [0.38, -0.24]
        ]
        flipper_points = [
            [[0.28, 0.28 + 0.34], [0.26, 0.26]],
            [[0.28, 0.28 + 0.34], [-0.26, -0.26]],
            [[-0.28, -0.28 - 0.34], [0.26, 0.26]],
            [[-0.28, -0.28 - 0.34], [-0.26, -0.26]],
        ]
        robot_points = torch.tensor(robot_points + [robot_points[0]])
        for points in flipper_points:
            plt.plot(points[1], points[0])
        plt.plot(robot_points[:, 1], robot_points[:, 0])
        plt.scatter(x=0, y=0)

        if hasattr(self, 'contacts'):
            for p in self.contacts:
                plt.scatter(x=p[1], y=p[0])

    def init_graphics_view(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        graphicscene = QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        graphicscene.addWidget(self.canvas)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        self.ui.graphicsView.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
        self.ui.graphicsView.show()  # 最后，调用show方法呈现图形！
