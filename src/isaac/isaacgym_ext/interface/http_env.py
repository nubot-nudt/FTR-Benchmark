import pickle
from datetime import datetime
from typing import Optional, Union, Tuple

import gym
import requests
import torch

from bottle import Bottle, request
app = Bottle()

from .numpy_env import NumpyAloneEnv

class HttpAloneServerEnv(NumpyAloneEnv):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.host = '0.0.0.0'
        self.port = 12346
        self.init_http_server()

    def init_http_server(self):
        @app.post('/step')
        def set_action():
            action = pickle.loads(request.body.read())
            ret = self.step(action)
            return pickle.dumps(ret)

        @app.get("/reset")
        def reset_info():
            obs = self.reset()
            return pickle.dumps(obs)

        @app.get('/info')
        def get_info():
            return pickle.dumps({
                'observation_space': self.observation_space,
                'action_space': self.action_space,
            })

    def run(self):
        app.run(host=self.host, port=self.port)

import pickle
import gym
import requests

class HttpClientEnv(gym.Env):

    def __init__(self, base_url='http://127.0.0.1:12346'):
        self.base_url = base_url
        self.session = requests.Session()

        content = self.session.get(f'{self.base_url}/info').content
        info_dict = pickle.loads(content)
        self.observation_space = info_dict['observation_space']
        self.action_space = info_dict['action_space']

    def step(self, action):
        content = self.session.post(f'{self.base_url}/step', data=pickle.dumps(action)).content
        ret = pickle.loads(content)
        return ret

    def reset(self):
        content = self.session.get(f'{self.base_url}/reset').content
        obs = pickle.loads(content)
        return obs


