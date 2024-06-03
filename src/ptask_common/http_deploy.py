import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from ptask_common.utils import apply_project_directory
apply_project_directory()
# --------------------------------------------------------------------------------

import numpy as np
import torch
import pickle
from gym import spaces

from bottle import Bottle, request
app = Bottle()

from sarl.algorithms.rl.sac.module import MLPActorCritic

from ptask_envs.envs.start import load_config_by_args
from ptask_common.utils import deploy_port
from ptask_common.utils import select_sarl_checkpoint_path
from ptask_common.utils.tensor import to_tensor

from ptask_rlgames.loader import SACRunsModelLoader

def deploy_sarl():
    path = select_sarl_checkpoint_path()
    cfg = load_config_by_args(['train=BM_SAC'])
    learn_cfg = cfg['train']['params']['learn']
    ac_kwargs = dict(hidden_sizes=[learn_cfg["hidden_nodes"]] * learn_cfg["hidden_layer"])

    action_space = spaces.Box(
        np.ones(4, dtype=np.float32) * -1.0, np.ones(4, dtype=np.float32) * 1.0
    )

    observation_space = spaces.Box(
        np.ones(121, dtype=np.float32) * -np.Inf,
        np.ones(121, dtype=np.float32) * np.Inf,
    )

    actor_critic = MLPActorCritic(observation_space, action_space, **ac_kwargs).to('cpu')
    actor_critic.load_state_dict(torch.load(path))
    actor_critic.eval()

    @app.post('/act')
    def act():

        obs = pickle.loads(request.body.read())
        obs = to_tensor(obs).to(torch.float32)
        print(obs)
        action = actor_critic.act(obs, deterministic=True)
        return pickle.dumps(action.numpy())

def deploy_rlgames():
    loader = SACRunsModelLoader('runs/Crossing_QVecSAC_diff', -1)
    # loader = SACRunsModelLoader('runs/Crossing_RlgamesSAC_batten', -1)

    @app.post('/act')
    def act():

        obs = pickle.loads(request.body.read())
        obs = to_tensor(obs).to(torch.float32)
        action = loader.get_action(obs)
        print(action)
        return pickle.dumps(action.numpy()[-4:])

if __name__ == '__main__':
    # deploy_sarl()
    deploy_rlgames()
    app.run(host='0.0.0.0', port=deploy_port)