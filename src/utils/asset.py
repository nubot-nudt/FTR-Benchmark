import os

asset_config_path = 'assets/config'

def get_asset_file_names() -> dict:
    return {i: f'{asset_config_path}/{i}' for i in sorted(os.listdir(asset_config_path))}