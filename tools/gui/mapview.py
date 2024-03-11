import os
import yaml
import threading

import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QWidget, QGraphicsScene

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT


from .ui.mapview import Ui_Form

from pumbaa.helper.map import MapHelper
from pumbaa.common import *
from utils.common.asset import get_asset_file_names

selected_file = ''
is_launch_cmd = False

class MapView(QWidget):

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.setWindowTitle('map')

        self.init_graphics_view()
        self.init_list_view()
        self.init_btn()


    def make_map(self):
        name =self.ui.listWidget.currentItem().text()
        filename = self.file_names[name]
        self.ui.make_map.setDisabled(True)

        self.print_text(f'开始制作地图: {filename}')

        with open('data/gui.yaml', 'r') as f:
            python_path = yaml.safe_load(f)['python']

        os.system(f'{python_path} src/generate_map.py {filename}')

        self.ui.make_map.setEnabled(True)
        self.print_text(f'地图制作完成: {filename}')
        self.update_map(name)


    def print_text(self, text):
        self.ui.textBrowser.append(text)
    def init_btn(self):
        def make_map_onclick():
            self.ui.textBrowser.clear()
            t = threading.Thread(target=self.make_map)
            t.daemon = True
            t.start()

        self.ui.make_map.clicked.connect(make_map_onclick)

        def start():
            global is_launch_cmd
            is_launch_cmd = True
            self.close()

        self.ui.start_play.clicked.connect(start)

    def init_list_view(self):
        self.file_names = get_asset_file_names()
        self.name_to_index_maps = dict(zip(self.file_names.keys(), range(len(self.file_names))))
        self.ui.listWidget.addItems(self.file_names.keys())

        self.ui.listWidget.currentTextChanged.connect(self.update_map)
        self.ui.listWidget.setCurrentRow(0)

    def update_map(self, name):
        global selected_file
        selected_file = self.file_names[name]
        asset = AssetEntry(file=selected_file)
        try:
            map_helper = MapHelper(**asset.map, rebuild=False)
        except FileNotFoundError:
            self.print_text('地图不存在,请制作地图')
            return

        plt.cla()
        plt.imshow(map_helper.map)
        self.canvas.draw()
    def init_graphics_view(self):
        self.fig = plt.figure()
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.fig_toolbar = NavigationToolbar2QT(self.canvas, self)
        graphicscene = QGraphicsScene()                 # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
        graphicscene.addWidget(self.canvas)             # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        # graphicscene.addWidget(self.fig_toolbar)
        self.ui.graph_layout.addWidget(self.fig_toolbar)
        self.fig_toolbar.setVisible(True)
        self.ui.graphicsView.setScene(graphicscene)     # 第五步，把QGraphicsScene放入QGraphicsView
        self.ui.graphicsView.show()                     # 最后，调用show方法呈现图形！