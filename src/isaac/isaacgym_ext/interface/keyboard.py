import pickle

import gym
import torch
from expiringdict import ExpiringDict

import threading
from bottle import Bottle, request
app = Bottle()

from isaacgym_ext.wrap.base import IsaacGymEnvWrapper
from utils.net import keyboard_port

from utils.tensor import to_numpy

class KeyboardEnv(IsaacGymEnvWrapper):

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.init_http_server()

        self.cmd_cache = ExpiringDict(max_len=100, max_age_seconds=1)


    def run(self):
        self.obs = self.reset()

        while self.env.simulation_app.is_running():

            actions = self.cmd_cache.get('cmd', {
                'vel_type': 'std',
                'vels': torch.tensor([0, 0]),
                'flipper_type': 'dt',
                'flippers': torch.tensor([0, 0, 0, 0])
            })

            ret = self.step(actions)
            if len(ret) == 5:
                self.obs, rewards, dones, _, self.info = ret
            else:
                self.obs, rewards, dones, self.info = ret

            if dones == True or self.reset_flag == True:
                self.reset_flag = False
                self.cmd = None
                self.reset()

    def init_http_server(self):
        self.ai_flipper = False

        @app.post('/set_ai_flipper')
        def set_ai_flipper():
            data = pickle.loads(request.body.read())
            self.ai_flipper = data

        @app.post('/set_action')
        def set_action():
            data = pickle.loads(request.body.read())
            self.cmd_cache['cmd'] = data


        self.reset_flag = False

        @app.get("/reset")
        def reset_info():
            self.reset_flag = True

        self.obs = None

        @app.get('/obs')
        def get_obs():
            if self.obs is not None:
                return pickle.dumps(self.obs)
            return dict()

        @app.get('/info')
        def get_info():
            if hasattr(self, 'info') and self.info is not None:
                return pickle.dumps(self.info)
            return dict()

        self.http_server = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=keyboard_port))
        self.http_server.daemon = True
        self.http_server.start()



