
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from utils.common.task import *


from PyQt5.QtWidgets import QApplication

from gui.trainview import TrainConfigView


import sys, os, shutil
import yaml

from utils.shell import execute_command

def exec_cmd(cfg, current_text):
    if current_text == 'op':
        cmd_list = [cfg['python'], 'scripts/adaptor_play.py']
    else:
        cmd_list = [cfg['python'], 'scripts/train_rlgames.py']

    key_arg_maps = [
        ['task', 'task'],
        ['train', 'train'],
        ['epoch', 'max_iterations'],
        ['headless', 'headless'],
        ['checkpoint', 'checkpoint'],
        ['asset', 'task.asset'],
        ['horizon', 'train.params.config.horizon_length'],
        ['minibatch', 'minibatch_size'],
        ['num_envs', 'num_envs'],
        ['action_mode', 'task.task.actionMode'],
    ]

    config = cfg[current_text]

    for key, arg in key_arg_maps:
        if key not in config:
            continue

        value = config[key]
        if isinstance(value, str):
            if value == '':
                continue
            cmd_list.append(f"{arg}='{value}'")
        elif isinstance(value, bool):
            cmd_list.append(f"{arg}={value}")
        elif isinstance(value, int):
            if value == 0:
                continue
            cmd_list.append(f"{arg}={value}")
        else:
            cmd_list.append(f"{arg}={value}")

    if current_text == 'train':
        cmd_list.append("test=False")

        if config['experiment'] is not None and config['experiment'] != '':
            experiment = config['task'] + \
                         ('' if config['train'].startswith(config['task']) else f'_{config["train"]}') + \
                         '_' + config['experiment']
        else:
            experiment = config['task']

        cmd_list.append(f"experiment={experiment}")

        if config['new'] == True:
            directory = f"runs/{experiment}"
            if config['backup'] == True:
                backup_task_nn(experiment)
            if os.path.exists(directory):
                shutil.rmtree(directory)

    elif current_text == 'play':
        cmd_list.append("test=True")
        cmd_list.append("headless=False")

    print(' '.join(cmd_list))
    print('=' * 80)
    print(' \\\n    '.join(cmd_list))

    if cfg['start'] == True:
        execute_command(' '.join(cmd_list))


if __name__ == '__main__':
    config_path = 'data/gui.yaml'
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    app = QApplication(sys.argv)

    main_view = TrainConfigView(cfg, 'train' if len(sys.argv) <= 1 else sys.argv[1])
    main_view.show()

    app.exec()

    with open(config_path, 'w') as f:
        yaml.dump(cfg, f)

    from gui.trainview import is_launch_cmd, current_text
    if is_launch_cmd == True:
        exec_cmd(cfg, current_text)