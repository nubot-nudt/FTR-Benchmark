from PyQt5.QtWidgets import QWidget, QFileDialog, QDesktopWidget

from .ui.trainview import Ui_Form

from processing.action_mode.define import ActionModeFactory
from utils.common.task import *

is_launch_cmd = False
current_text = ''
class TrainConfigView(QWidget):

    def __init__(self, cfg, default_tab='train'):
        super().__init__()
        self.ui = Ui_Form()
        self.cfg = cfg
        self.ui.setupUi(self)
        self.setWindowTitle('GUI_START')

        screen = QDesktopWidget().screenGeometry()  # 获取屏幕的大小
        top = int((screen.width() - self.width()) / 2)  # 顶部坐标
        left = int((screen.height() - self.height()) / 2)  # 左侧坐标
        self.setGeometry(top - 100, left - 100, self.width(), self.height())  # 设置居中

        self.tab_text = ['train', 'play', 'op']

        self.init_common_view()

        self.init_train_view()
        self.init_play_view()
        self.init_op_view()

        self.ui.tabWidget.setCurrentIndex(self.tab_text.index(default_tab))

        self.ui.start_pushButton.clicked.connect(self.start_on_click)
        self.ui.cmd_pushButton.clicked.connect(self.cmd_on_click)

    def init_common_view(self):
        self.init_combox_task_and_train()

    def init_combox_task_and_train(self):
        self.tasks = get_task_names()
        self.trains = get_train_names()

        for tab in self.tab_text:
            cfg = self.cfg[tab]

            # 初始化task
            if hasattr(self.ui, f'{tab}_task'):
                task_box = getattr(self.ui, f'{tab}_task')
                task_box.addItems(self.tasks)

                if cfg['task'] is not None and cfg['task'] != '':
                    task_box.setCurrentText(cfg['task'])

            # 初始化train
            if hasattr(self.ui, f'{tab}_train') and hasattr(self.ui, f'{tab}_task'):
                train_box = getattr(self.ui, f'{tab}_train')
                train_box.addItems(self.trains)

                if cfg['train'] is not None and cfg['train'] != '':
                    train_box.setCurrentText(cfg['train'])
                elif task_box.currentText() + "PPO" in self.trains:
                    train_box.setCurrentText(task_box.currentText() + "PPO")

            # 初始化task和train之间的关系
            if hasattr(self.ui, f'{tab}_train') and hasattr(self.ui, f'{tab}_task'):
                def on_train_text_changed(text):
                    attr_ = self.current_text + '_train'
                    if not hasattr(self.ui, attr_):
                        return
                    train_box = getattr(self.ui, attr_)

                    if text + "PPO" in self.trains:
                        train_box.setCurrentText(task_box.currentText() + "PPO")
                    for train_text in self.trains:
                        if train_text.startswith(text):
                            train_box.setCurrentText(train_text)

                task_box.currentTextChanged.connect(on_train_text_changed)

            # 初始化action_mode
            if hasattr(self.ui, f'{tab}_action_mode'):
                action_mode_box = getattr(self.ui, f'{tab}_action_mode')
                action_mode_list = [''] + ActionModeFactory.get_names()
                action_mode_box.addItems(action_mode_list)
                if cfg['action_mode'] is not None and cfg['action_mode'] != '':
                    action_mode_box.setCurrentText(cfg['action_mode'])

    def init_op_view(self):
        cfg = self.cfg['op']

        self.ui.op_checkpoint.setText(cfg['checkpoint'])
        self.ui.op_asset.setText(cfg['asset'])

        self.ui.op_load.clicked.connect(lambda : self.load_file(self.ui.op_checkpoint))
        self.ui.op_select.clicked.connect(lambda :self.asset_select(self.ui.op_asset))

    def init_train_view(self):

        cfg = self.cfg['train']

        self.ui.train_checkpoint.setText(cfg['checkpoint'])
        self.ui.train_headless.setChecked(cfg['headless'])
        self.ui.train_new.setChecked(cfg['new'])
        self.ui.train_backup.setChecked(cfg['backup'])
        self.ui.train_asset.setText(cfg['asset'])
        self.ui.train_experiment.setText(cfg['experiment'])
        self.ui.train_epoch.setValue(cfg['epoch'])
        self.ui.train_num_envs.setValue(cfg['num_envs'])

        def new_on_change(is_check):
            self.ui.train_checkpoint.clear()

            disable_list = [self.ui.train_checkpoint, self.ui.train_load]
            enable_list = [self.ui.train_backup]
            for w in disable_list:
                if is_check:
                    w.setDisabled(True)
                else:
                    w.setEnabled(True)
            for w in enable_list:
                if is_check:
                    w.setEnabled(True)
                else:
                    w.setDisabled(True)

        new_on_change(self.ui.train_new.isChecked())
        self.ui.train_new.toggled.connect(new_on_change)

        self.ui.train_load.clicked.connect(lambda : self.load_file(self.ui.train_checkpoint))
        self.ui.train_select.clicked.connect(lambda :self.asset_select(self.ui.train_asset))

        def set_value_default():
            self.ui.train_num_envs.setValue(0)
            self.ui.train_horizon.setValue(0)
            self.ui.train_minibatch.setValue(0)

        self.ui.train_use_default.clicked.connect(set_value_default)

        def value_change():
            a = self.ui.train_num_envs.value()
            b = self.ui.train_horizon.value()

            self.ui.train_minibatch.setRange(0, a * b)
            self.ui.train_minibatch.setValue(a * b)

        self.ui.train_num_envs.valueChanged.connect(value_change)
        self.ui.train_horizon.valueChanged.connect(value_change)
    def init_play_view(self):

        cfg = self.cfg['play']

        self.ui.play_checkpoint.setText(cfg['checkpoint'])
        self.ui.play_asset.setText(cfg['asset'])
        self.ui.play_num_envs.setValue(cfg['num_envs'])

        self.ui.play_load.clicked.connect(lambda : self.load_file(self.ui.play_checkpoint))
        self.ui.play_select.clicked.connect(lambda :self.asset_select(self.ui.play_asset))

    def cmd_on_click(self):
        self.cfg['start'] = False
        self.save_and_exit()

    def start_on_click(self):
        self.cfg['start'] = True
        self.save_and_exit()

    def save_and_exit(self):
        global is_launch_cmd, current_text
        is_launch_cmd = True
        current_text = self.current_text

        config_field_function_maps = [
            ['task', 'task', 'currentText'],
            ['train', 'train', 'currentText'],
            ['checkpoint', 'checkpoint', 'toPlainText'],
            ['headless', 'headless', 'isChecked'],
            ['new', 'new', 'isChecked'],
            ['backup', 'backup', 'isChecked'],
            ['asset', 'asset', 'text'],
            ['num_envs', 'num_envs', 'value'],
            ['horizon', 'horizon', 'value'],
            ['minibatch', 'minibatch', 'value'],
            ['experiment', 'experiment', 'text'],
            ['epoch', 'epoch', 'value'],
            ['action_mode', 'action_mode', 'currentText'],

        ]
        for cfg, field, func in config_field_function_maps:
            if not hasattr(self.ui, f'{current_text}_{field}'):
                continue
            print(cfg)
            self.cfg[current_text][cfg] = getattr(getattr(self.ui, f'{current_text}_{field}'), func)()


        self.close()

    def load_file(self, text_edit):

        if text_edit.toPlainText() != '':
            path = os.path.dirname(text_edit.toPlainText())
        else:
            attr_ = f'{self.current_text}_task'
            path = f'runs/{getattr(self.ui, attr_).currentText()}/nn'


        if not os.path.exists(path):
            path = f'runs'

        filename = QFileDialog.getOpenFileName(self, directory=path)
        text_edit.setText(filename[0])

    def asset_select(self, line_edit):
        filename = QFileDialog.getOpenFileName(self, directory=f'assets/config')
        line_edit.setText(filename[0])
    @property
    def current_text(self):
        return self.tab_text[self.ui.tabWidget.currentIndex()]



