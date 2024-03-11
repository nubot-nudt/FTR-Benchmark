import sys
from pprint import pprint

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import fire

from pipetools import pipe, X
from pipetools.utils import debug_print

class ExperimentPlot():

    def metrics(self, output_type='pprint'):
        df = pd.read_csv('./data/experiment/metrics')

        df['yaw'] = df['yaw'].apply(lambda x: x + 2 * np.pi if x < 0 else x)


        dt = df['time'].diff().mode().iloc[0]

        mark = {
            'APS': pipe | X['pitch'].diff().abs().mean() / dt,
            'APS_2': pipe | X['ang_pitch'].abs().mean(),

            'ARS': pipe | X['roll'].diff().abs().mean() / dt,
            'ARS_2': pipe | X['ang_roll'].abs().mean(),

            'AHS': pipe | X['z'].diff().abs().mean() / dt,
            'AHS_2': pipe | X['lin_z'].abs().mean(),

            'MAS': pipe | X[['pitch', 'roll', 'yaw']].diff().abs().max().sum() / dt,
            'MAS_2': pipe | X[['ang_pitch', 'ang_roll', 'ang_yaw']].abs().max().sum(),

            'MLS': pipe | X[['x', 'y', 'z']].diff().diff().abs().max().sum() / dt / dt,
            'MLS_2': pipe | X[['lin_x', 'lin_y', 'lin_z']].diff().abs().max().sum() / dt,

            't_cost': pipe | X['time'].max(),
        }

        ret = {k:list() for k in mark}

        for i, g in df.groupby('epoch'):
            for k, f in mark.items():
                ret[k].append(f(g))

        t = dict()
        for k, d in ret.items():
            t[k] = {
                'mean': np.mean(d),
                'std': np.std(d),
            }

        if output_type == 'latex':
            for k in mark:
                print(k, end=' & ')
            print()
            for k in mark:
                print('{:.3f}$\pm${:.3f}'.format(t[k]['mean'], t[k]['std']), end=' & ')
            print()
        else:
            pprint(t)

    def plot_trajectory(self, epoch=1, block=True):
        df = pd.read_csv('./data/experiment/metrics')
        df = df[df['epoch'] == epoch]

        plt.plot(-df['x'], df['z'])
        plt.show(block=block)

    def reward(self, epoch=1, trajectory=False):

        if trajectory:
            self.plot_trajectory(block=False)

        df = pd.read_csv('./data/experiment/reward')
        names = df['name'].drop_duplicates()

        fig = plt.figure(figsize=(16, 12))
        for i, name in enumerate(filter(lambda x: x != 'end', names)):
            plt.subplot(3, len(names) // 3+1, i + 1)
            data = df[(df['name'] == name) & (df['epoch'] == epoch)]
            sns.lineplot(data, x='step', y='reward', label=name)
        plt.legend()
        plt.show(block=False)

        fig = plt.figure(figsize=(8, 6))
        for i, name in enumerate(filter(lambda x: x != 'end', names)):
            data = df[(df['name'] == name) & (df['epoch'] == epoch)].copy()
            data['coef_reward'] = data['reward'] * df['coef']
            data['acc_reward'] = data['coef_reward'].cumsum()
            sns.lineplot(data, x='step', y='acc_reward', label=name)
        plt.legend()
        plt.show()


if __name__ == '__main__':
    fire.Fire(ExperimentPlot)