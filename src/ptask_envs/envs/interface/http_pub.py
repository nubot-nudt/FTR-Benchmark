import pickle

import threading
from bottle import Bottle

app = Bottle()

from ptask_envs.envs.wrap.base import IsaacGymEnvWrapper
from ptask_common.utils.net import pub_port
from ptask_common.utils.tensor import to_numpy

class HttpPubEnv(IsaacGymEnvWrapper):

    _name_to_field_maps = {
        'pos': 'positions',
        'orient': 'orientations_3',
        'flippers': 'flipper_positions',
        'vels': 'velocities',
        'map': 'current_frame_height_maps',
    }

    def __init__(self, env):
        super().__init__(env)
        self.init_http_server()
        self.all_info_dict = dict()

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        for name, field in self._name_to_field_maps.items():
            self.all_info_dict[name] = to_numpy(getattr(self.task, field))


        return obs, rew, done, info

    def init_http_server(self):

        @app.get('/all_info')
        def all_info():
            return pickle.dumps(self.all_info_dict)


        self.http_server = threading.Thread(target=lambda: app.run(host='0.0.0.0', port=pub_port))
        self.http_server.setDaemon(True)
        self.http_server.start()



