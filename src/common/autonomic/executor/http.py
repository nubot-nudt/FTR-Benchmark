import pickle
import requests

from autonomic.common.base import BaseExecutor
from utils.net import deploy_port


class HttpPredictor(BaseExecutor):

    def get_action(self, obs):
        resp = requests.post(f'http://127.0.0.1:{deploy_port}/act', data=pickle.dumps(obs))
        return pickle.loads(resp.content)

    def predict(self, *args, **kwargs):
        pass

