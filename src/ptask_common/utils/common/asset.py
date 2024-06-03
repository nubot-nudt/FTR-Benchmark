import os
import re
import hashlib

from loguru import logger

from tinydb import TinyDB, Query
from pipetools import pipe

from ptask_common.utils.shell import execute_command
from ptask_common import isaac_python_path

asset_config_path = 'assets/config'
asset_map_path = 'assets/map'
asset_db_file = 'assets/height_mao_db.json'

db = TinyDB(asset_db_file)
MapEntry = Query()


def get_asset_config_file_path_by_name(name):
    name = name.removesuffix('.yaml')
    return f'{asset_config_path}/{name}.yaml'


def get_map_file_path_by_name(name):
    name = name.removesuffix('.yaml')
    return f'{asset_map_path}/{name}.map'


def _md5_asset_config(name):
    if not name.endswith('yaml'):
        name += '.yaml'

    with open(f'{asset_config_path}/{name}') as f:
        text = (pipe
                | (map, lambda x: x.strip())
                | (filter, lambda x: not (x.startswith('#') or len(x) == 0))
                | ''.join
                )(f.readlines())
        text = re.sub(r'\s', '', text)

    md5 = hashlib.md5()
    md5.update(text.encode())
    return md5.hexdigest()


def update_lastest_map_config(name):
    update_map_config_md5(name, _md5_asset_config(name))


def check_lastest_map(name):
    if not os.path.exists(get_map_file_path_by_name(name)):
        return False

    elif get_map_config_md5(name) != _md5_asset_config(name):
        return False

    else:
        return True


def check_and_map(file):
    name = os.path.basename(file).removesuffix('.yaml')

    if not check_lastest_map(name):
        logger.warning(f'{name}\'s map is not found to be under construction')
        execute_command(
            f'{isaac_python_path} src/generate_map.py {get_asset_config_file_path_by_name(name)} >> /dev/null 2>&1')
        update_lastest_map_config(name)


def update_map_config_md5(name, md5):
    is_exist = db.search(MapEntry.name == name)

    if is_exist:
        db.update({'md5': md5}, MapEntry.name == name)
    else:
        db.insert({'name': name, 'md5': str(md5)})


def get_map_config_md5(name):
    result = db.search(MapEntry.name == name)

    if len(result) == 0:
        return ''
    else:
        return result[0]['md5']


def get_asset_pure_names():
    return [i.removesuffix('.yaml') for i in sorted(os.listdir(asset_config_path))]


# def get_asset_names():
#     return [i for i in sorted(os.listdir(asset_config_path))]


def get_asset_file_names() -> dict:
    return {i: f'{asset_config_path}/{i}' for i in sorted(os.listdir(asset_config_path))}
