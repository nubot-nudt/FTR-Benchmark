import os
import pickle

import requests

from . import keyboard_port, pub_port

base_url = os.getenv("PTASK_SERVER_HOST")
if base_url is None or len(base_url) == 0:
    base_url = '127.0.0.1'
base_url = f'http://{base_url}'


def set_keyboard_position(pos):
    requests.post(f'{base_url}:{keyboard_port}/set_position', data=pickle.dumps(pos))


def get_keyboard_position():
    resp = requests.get(f'{base_url}:{keyboard_port}/current_position')
    return pickle.loads(resp.content)


def get_keyboard_obs():
    obs = requests.get(f'{base_url}:{keyboard_port}/obs').content
    return pickle.loads(obs)


def get_pub_info():
    info = requests.get(f'{base_url}:{pub_port}/all_info').content
    return pickle.loads(info)
