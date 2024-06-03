import pickle
import sys

import gym
import torch
from expiringdict import ExpiringDict

import threading
from flask import Flask, request

app = Flask(__name__)
import logging
app.logger.setLevel(logging.CRITICAL)

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper
from ptask_common.utils.net import keyboard_port


class KeyboardEnv(IsaacGymEnvWrapper):

    def __init__(self, env: gym.Env, config_path):
        super().__init__(env)
        self.config_path = config_path

        self.init_http_server()

        self.cmd_cache = ExpiringDict(max_len=100, max_age_seconds=1)

    def run(self):
        self.obs = self.reset()

        try:
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
        except Exception:
            import traceback
            traceback.print_exc(file=sys.stdout)
            sys.exit(1)

    def init_http_server(self):
        self.ai_flipper = False

        @app.post('/set_ai_flipper')
        def set_ai_flipper():
            data = pickle.loads(request.get_data())
            self.ai_flipper = data

        @app.post('/set_action')
        def set_action():
            data = pickle.loads(request.get_data())
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

        @app.get('/config_path')
        def get_config_path():
            return self.config_path

        @app.get('/current_position')
        def get_current_position():
            return pickle.dumps(self.task.position)

        @app.post('/set_position')
        def set_position():
            pos = pickle.loads(request.get_data())
            self.task.set_robot_position(pos)

        self.http_server = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=keyboard_port))
        self.http_server.daemon = True
        self.http_server.start()
