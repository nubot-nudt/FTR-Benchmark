import yaml
import pickle
import os

default_birth_maps_file = './data/reset_info_maps.yaml'


def save(birth_file, birth_info):
    with open(birth_file, 'wb') as f:
        pickle.dump(birth_info, f)


def load_birth_file_and_info(asset, birth_maps_file=default_birth_maps_file):
    with open(birth_maps_file, 'r') as maps_stream:
        birth_file = yaml.safe_load(maps_stream)[asset]

    with open(birth_file, 'rb') as f:
        birth_info = pickle.load(f)

    return birth_file, birth_info


def load_birth_info(asset, birth_maps_file=default_birth_maps_file):
    _, birth_info = load_birth_file_and_info(asset, birth_maps_file)
    return birth_info
