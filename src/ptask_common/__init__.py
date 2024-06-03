# -*- coding: utf-8 -*-
"""
====================================
@File Name ：__init__.py.py
@Time ： 2024/5/21 下午7:27
@Program IDE ：PyCharm
@Create by Author ： hongchuan zhang
====================================

"""
import os
import traceback
from loguru import logger


def init_logger():
    stack_trace = traceback.extract_stack()
    bottom_frame = stack_trace[0]
    launched_file = bottom_frame.filename
    name = os.path.basename(launched_file)[:-3]

    log_file = f'logs/run_log_{name}.log'

    logger.add(log_file, rotation='10MB', retention='72h', encoding='utf-8')


def get_project_root_directory():
    current_dir = os.path.abspath(os.path.dirname(__file__))

    check_subdirectories = ['src', 'cfg']

    while current_dir != '/':
        if all([os.path.exists(os.path.join(current_dir, d)) for d in check_subdirectories]):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    else:
        raise RuntimeError('You are not within the scope of the project.')


isacc_sim_root_path = os.popen('realpath ~/.local/share/ov/pkg/isaac_sim*').read().strip()
project_dir = get_project_root_directory()
isaac_python_path = os.path.join(project_dir, 'python.sh')


def project_dir_join(path):
    return os.path.join(project_dir, path)


def apply_project_directory():
    stack_trace = traceback.extract_stack()
    bottom_frame = stack_trace[0]
    # launched_file = bottom_frame.filename
    # launched_file_dir = os.path.dirname(launched_file)

    # os.chdir(project_dir)

    # 设置源码目录
    # sys.path.append(os.path.join(project_dir, launched_file_dir))
    # for d in os.listdir('src'):
    #     sys.path.append(os.path.join(project_dir, f'src/{d}'))
    #
    # if os.path.basename(launched_file) != 'tools':
    #     sys.path.append(os.path.join(project_dir, 'src_isaac'))

    # 导入zip格式的库文件
    # for zip_file in filter(lambda x: x.endswith('.zip'), os.listdir('deps')):
    #     sys.path.append(os.path.join(os.path.join(project_dir, 'deps'), zip_file))

    init_logger()
