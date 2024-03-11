import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()
# --------------------------------------------------------------------------------


import pickle
from time import sleep

import requests
base_url = 'http://127.0.0.1:12345'

from processing.robot_info.base import RobotState
from autonomic.predictor.flipper import FlipperPredictor
from autonomic.executor.rlgames import RlgamesPredictor


def runs_pos_control():
    predictor = FlipperPredictor()

    while True:
        obs = requests.get(f'{base_url}/obs').content
        state = RobotState(pickle.loads(obs))
        flippers = predictor(state)

        print(state.vels)

        # flipper_points = robot_flipper_positions(state.flipper)
        # flipper_points = robot_to_world(state.roll, state.pitch, flipper_points)
        # print(flipper_points)

        requests.post(f'{base_url}/set_action', data=pickle.dumps({
            'flipper_type': 'pos_dt',
            'flippers': flippers,
        }))
        sleep(0.1)

def runs_robot_state_control(path):
    predictor = RlgamesPredictor.load_model(path, 124, [3, 3, 3, 3])
    # predictor = PosePredictor()

    while True:
        obs = requests.get(f'{base_url}/obs').content
        state = RobotState(pickle.loads(obs))
        flippers = predictor(state)

        requests.post(f'{base_url}/set_action', data=pickle.dumps({
            'flipper_type': 'dt',
            'flippers': flippers,
        }))
        sleep(0.1)
def runs_origin_obs_control(path):
    predictor = RlgamesPredictor.load_model(path, 3000)
    # predictor = PosePredictor()

    while True:
        obs = requests.get(f'{base_url}/obs').content
        obs = pickle.loads(obs)
        obs = obs['obs']
        flippers = (predictor.get_action(obs) - 1) * 2

        requests.post(f'{base_url}/set_action', data=pickle.dumps({
            'flipper_type': 'dt',
            'flippers': flippers,
        }))
        sleep(0.1)

def main():
    runs_origin_obs_control('runs/LowerHier_RlgamesSAC_sdown_22-17-32-19')
    # runs_robot_state_control('runs/LowerHier_MlpExtractorPPO_stick-batten')
    # runs_pos_control()

if __name__ == '__main__':
    main()