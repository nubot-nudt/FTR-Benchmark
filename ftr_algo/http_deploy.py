import numpy as np
import torch
import pickle
from gym import spaces
import sys

from flask import Flask, request

app = Flask(__name__)

def to_tensor(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)

    if isinstance(data, torch.Tensor):
        return data

    return torch.tensor(data)

    data = np.asarray(data)
    data = torch.from_numpy(data).float()
    return data

from ftr_algo.executor import load_rl

def deploy_ftr():
    path = sys.argv[1]
    executor = load_rl(checkpoint=path)

    @app.post('/act')
    def act():
        obs = pickle.loads(request.data)
        obs = to_tensor(obs).to(torch.float32)
        print(obs)
        action = executor(obs)
        return pickle.dumps(action.numpy())


if __name__ == '__main__':
    deploy_ftr()
    app.run(host='0.0.0.0', port=12344)
