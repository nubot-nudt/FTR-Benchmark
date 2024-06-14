import yaml
import pickle
import os

default_birth_maps_file = './data/reset_info_maps.yaml'


def save_to_assets(asset, birth_info):
    asset_name = asset.removesuffix('.yaml')
    birth_file = f'assets/birth/{asset_name}'
    with open(birth_file, 'wb') as f:
        pickle.dump(birth_info, f)


def save(birth_file, birth_info):
    with open(birth_file, 'wb') as f:
        pickle.dump(birth_info, f)


def load_birth_file_and_info(asset, birth_maps_file=default_birth_maps_file):
    asset_name = asset.removesuffix('.yaml')

    # 尝试从默认映射文件中找
    if os.path.exists(default_birth_maps_file):
        with open(birth_maps_file, 'r') as maps_stream:
            birth_file = yaml.safe_load(maps_stream).get(asset, None)

    # 尝试从assets/birth找
    if 'birth_file' not in locals():
        if os.path.exists(f'assets/birth/{asset_name}'):
            birth_file = f'assets/birth/{asset_name}'

    if 'birth_file' in locals():
        # 找到了
        with open(birth_file, 'rb') as f:
            birth_info = pickle.load(f)
    else:
        # 尝试从asset文件的task_info找
        with open(f'assets/config/{asset_name}.yaml') as f:
            asset_config = yaml.safe_load(f)

        if 'task_info' in asset_config:
            birth_info = [asset_config['task_info']]
        else:
            raise RuntimeError(f'could not solve birth_info for {asset_name}')

    return birth_file, birth_info


def load_birth_info(asset, birth_maps_file=default_birth_maps_file):
    _, birth_info = load_birth_file_and_info(asset, birth_maps_file)
    return birth_info
