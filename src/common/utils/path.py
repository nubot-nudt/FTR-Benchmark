import os
import sys
import traceback

from .log import init_logger

project_dir = ''

def project_dir_join(path):
    return os.path.join(project_dir, path)

def get_project_root_directory():
    current_dir = os.path.abspath(os.path.dirname(__file__))

    check_subdirectories = ['src/common', 'src/deps']

    while current_dir != '/':
        if all([os.path.exists(os.path.join(current_dir, d)) for d in check_subdirectories]):
            return current_dir
        current_dir = os.path.dirname(current_dir)
    else:
        raise RuntimeError('You are not within the scope of the project.')

def get_isacc_sim_root_path():
    return os.popen('realpath ~/.local/share/ov/pkg/isaac_sim*').read().strip()

def get_isaac_python_path():
    return os.path.join(get_isacc_sim_root_path(), 'python.sh')


def apply_project_directory():
    global project_dir
    project_dir = get_project_root_directory()
    stack_trace = traceback.extract_stack()
    bottom_frame = stack_trace[0]
    launched_file = bottom_frame.filename
    launched_file_dir = os.path.dirname(launched_file)

    os.chdir(project_dir)

    # 设置源码目录
    sys.path.append(os.path.join(project_dir, launched_file_dir))
    for d in os.listdir('src'):
        sys.path.append(os.path.join(project_dir, f'src/{d}'))

    if os.path.basename(launched_file) != 'tools':
        sys.path.append(os.path.join(project_dir, 'src_isaac'))

    # 导入zip格式的库文件
    # for zip_file in filter(lambda x: x.endswith('.zip'), os.listdir('deps')):
    #     sys.path.append(os.path.join(os.path.join(project_dir, 'deps'), zip_file))

    init_logger()

    # print(sys.path)
    # sys.exit()

#     return
#     '''
#     ['/home/zhc/.local/share/opt/ros/noetic/lib/python3/dist-packages', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.kit', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.gym', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/kernel/py', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/plugins/bindings-python', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.lula/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.exporter.urdf/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/exts/omni.kit.pip_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.core_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.ml_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.pip.compute/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.pip.cloud/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/extscache/omni.pip.torch-2_0_1-2.0.2+105.1.lx64/torch-2-0-1', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10/site-packages', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python310.zip', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10/lib-dynload', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/bindings-python', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/scripting-python-3.10/libs']
# ['/home/zhc/.local/share/opt/ros/noetic/lib/python3/dist-packages', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.kit', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.gym', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/kernel/py', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/plugins/bindings-python', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.lula/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.exporter.urdf/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/exts/omni.kit.pip_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.core_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.isaac.ml_archive/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.pip.compute/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/exts/omni.pip.cloud/pip_prebundle', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/extscache/omni.pip.torch-2_0_1-2.0.2+105.1.lx64/torch-2-0-1', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10/site-packages', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python310.zip', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10', '/home/zhc/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/lib/python3.10/lib-dynload', '/home/zhc/isaacsim_prj/pumbaa_task/scripts/../src', '/home/zhc/isaacsim_prj/pumbaa_task/scripts', '/home/zhc/isaacsim_prj/pumbaa_task/src', '/home/zhc/isaacsim_prj/pumbaa_task/deps', '/home/zhc/isaacsim_prj/pumbaa_task/src_isaac']
#     '''

    # module_list = [
    #     'kit/bindings-python',
    #     'kit/scripting-python-3.10/libs',
    #     # 'omni.isaac.ui',
    #    # 'omni.isaac.sensor',
    #     # 'omni.isaac.IsaacSensorSchema',
    #     # 'omni.usd.schema.isaac',
    #     # 'omni.isaac.core_nodes',
    #     # 'omni.isaac.range_sensor',
    # ]
    # module_root_path = f'{get_isacc_sim_root_path()}'
    #
    # for m in module_list:
    #     sys.path.append(os.path.join(module_root_path, m))
