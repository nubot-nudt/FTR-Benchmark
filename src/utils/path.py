import os
import sys
from .log import init_logger

project_dir = ''

def project_dir_join(path):
    return os.path.join(project_dir, path)

def get_project_root_directory():
    current_dir = os.path.abspath(os.path.dirname(__file__))

    check_subdirectories = ['src', 'deps']

    while current_dir != '/':
        if all([os.path.exists(os.path.join(current_dir, d)) for d in check_subdirectories]):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    else:
        raise RuntimeError('You are not within the scope of the project.')

def get_isacc_sim_root_path():
    return os.popen('realpath ~/.local/share/ov/pkg/isaac_sim*').read().strip()

def apply_project_directory():
    global project_dir
    project_dir = get_project_root_directory()
    os.chdir(project_dir)

    # 设置源码目录
    sys.path.append(os.path.join(project_dir, 'src'))
    sys.path.append(os.path.join(project_dir, 'deps'))

    # 导入zip格式的库文件
    for zip_file in filter(lambda x: x.endswith('.zip'), os.listdir('deps')):
        sys.path.append(os.path.join(os.path.join(project_dir, 'deps'), zip_file))

    init_logger()

    # module_list = [
    #     'omni.isaac.ui',
    #     'omni.isaac.sensor',
    #     'omni.isaac.IsaacSensorSchema',
    #     'omni.usd.schema.isaac',
    #     'omni.isaac.core_nodes',
    #     'omni.isaac.range_sensor',
    # ]
    # module_root_path = f'{get_isacc_sim_root_path()}/exts'
    #
    # for m in module_list:
    #     sys.path.append(os.path.join(module_root_path, m))
