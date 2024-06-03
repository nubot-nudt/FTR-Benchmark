from ptask_common.utils.common import asset, runs
from ptask_common.utils.tui.get_options import get_options


def select_asset_name():
    return get_options(asset.get_asset_file_names(), title='Please select asset', is_exit=True)

def select_asset_path():
    assets = asset.get_asset_file_names()
    name = get_options(asset.get_asset_file_names(), title='Please select asset', is_exit=True)
    return assets[name]

def select_sarl_checkpoint_path():
    experiment_name = get_options(runs.get_sarl_runs_names(), title='Please select experiment name', is_exit=True)
    checkpoint_name = get_options(runs.get_sarl_checkpoint_names(experiment_name), is_exit=True)
    return runs.get_sarl_checkpoint_abspath(experiment_name, checkpoint_name)