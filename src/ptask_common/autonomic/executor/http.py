import pickle
import requests

from ptask_common.autonomic.executor import BaseExecutor
from ptask_common.utils.net import deploy_port


class HttpPredictor(BaseExecutor):

    def get_action(self, obs):
        resp = requests.post(f'http://127.0.0.1:{deploy_port}/act', data=pickle.dumps(obs))
        return pickle.loads(resp.content)

    def predict(self, *args, **kwargs):
        pass

