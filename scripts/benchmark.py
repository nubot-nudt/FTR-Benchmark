
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()
# --------------------------------------------------------------------------------

import re
import shutil

from collections import defaultdict

import numpy as np
import glob
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, FuncFormatter
import seaborn as sns
from tqdm import tqdm
import pandas as pd
import fire
from tensorboard.backend.event_processing import event_accumulator

from utils.shell import execute_command, command_result
from utils.common import runs

class Benchmark:

    _algos = {
        'ppo': 'BM_PPO',
        'sac': 'BM_SAC',
        'ddpg': 'BM_DDPG',
        'td3': 'BM_TD3',
        'trpo': 'BM_TRPO',
    }

    _assets = {
        'steps': 'terrain_steps',
        'plum': 'benchmark_plum',
        'sdown': 'benchmark_sdown',
        'sup': 'benchmark_sup',
        'batten': 'benchmark_batten',
        'wave': 'terrain_wave',
        'mixed': 'map_mixed',
        'flat': 'benchmark_flat',
    }

    _seeds = [40, 50, 60]

    _max_iterations = 3000

    _save_dir = './data/benchmark'

    def train_algos(self, asset, seed):
        for algo in self._algos:
            self.train(algo, asset, seed)

    def train_assets(self, algo, seed):
        for asset in self._assets:
            self.train(algo, asset, seed)

    def train_seeds(self, algo, asset):
        for seed in self._seeds:
            self.train(algo, asset, seed)

    def train(self, algo, asset, seed):
        """
        Quickly launch a training program.
        :param algo: Specify the algorithm used for training, which can be ppo, sac, ddpg, td3 ,or trpo.
        :param asset: Specify the task scenario to be used.
        :param seed: The specified seed number can be 40, 50, or 60.
        :return:
        """

        if algo not in self._algos:
            print(f'{algo=} must be in {list(self._algos.values())}')
            sys.exit(1)

        if asset not in self._assets:
            print(f'{asset=} must be in {list(self._assets.values())}')
            sys.exit(1)

        seed = int(seed)
        if seed not in self._seeds:
            print(f'{seed=} must be in {self._seeds}')
            sys.exit(1)

        runs_dir = f'runs/bm_{asset}_{algo}_{seed}'

        if os.path.isdir(runs_dir):
            shutil.rmtree(runs_dir)

        cmd_str_list = [
            '~/.local/share/ov/pkg/isaac_sim-*/python.sh',
            'scripts/train_sarl.py',
            'task=Benchmark',
            f'train={self._algos[algo]}',
            f'max_iterations={self._max_iterations}',
            'headless=True',
            'num_envs=128',
            f'task.asset="./assets/config/{self._assets[asset]}.yaml"',
            f'seed={int(seed)}',
            'rl_device=cpu',
            'sim_device=cpu',
            f'experiment={runs_dir}'
        ]

        execute_command(' '.join(cmd_str_list))

    def play(self, checkpoint=None, asset=None, num_envs=5, follow_camera=None):

        if checkpoint is None:
            from tui.select_info import select_sarl_checkpoint_path
            checkpoint = select_sarl_checkpoint_path()

        checkpoint = os.path.abspath(checkpoint)
        checkpoint_dir = os.path.dirname(checkpoint)
        experiment_name = checkpoint_dir.split("/")[-1]

        _, _asset, algo, seed = experiment_name.split("_")

        if asset is None:
            asset = self._assets[_asset]

        if follow_camera is None:
            follow_camera = True if num_envs == 1 else False

        cmd_str_list = [
            '~/.local/share/ov/pkg/isaac_sim-*/python.sh',
            'scripts/train_sarl.py',
            'task=Benchmark',
            f'train={self._algos[algo]}',
            'headless=False',
            f'num_envs={num_envs}',
            'test=True',
            'debug=True',
            f'task.asset="./assets/config/{asset}.yaml"',
            f'seed={int(seed)}',
            'rl_device=cpu',
            'sim_device=cpu',
            f'checkpoint="{checkpoint}"',
            f'task.env.follow_camera={follow_camera}',
        ]

        execute_command(' '.join(cmd_str_list))
        execute_command('rm -rf runs/*_$(hostname)')

    def killall(self):
        """
        Kill the process being trained in use.
        """

        result = command_result(f'pgrep -f scripts/benchmark.py -a')
        for line in result.split('\n'):
            if f'{os.getpid()}' in line or 'killall' in line or 'pgrep' in line:
                continue
            pid = re.split(r'\s', line)[0]
            execute_command(f'kill -9 {pid}')

        execute_command('pkill -f scripts/train_sarl.py -9')



    def ps(self):
        """
        List the tasks currently being trained.
        """
        result = command_result('pgrep -f scripts/train_sarl.py -a')
        no = 0
        for line in result.split('\n'):
            if 'pgrep' in line or 'python3' not in line:
                continue
            no += 1
            for i, word in enumerate(re.split(r'\s', line)):
                if i == 0:
                    n = 30
                    print()
                    print('*' * n + f' {no} ' + '*' * n)
                    print(f'pid={word}')
                elif i == 1:
                    print(f'{word} \\')
                else:
                    print(f'\t{word} \\')


    def csv(self, *experiment_dirs):
        """
        Convert tensorboard output common to CSV.
        :param experiment_dirs: experiment_dirs, for example, runs/bm_steps_ppo_34
        :return:
        """
        experiment_dirs = runs.get_sarl_runs_dir()

        for d in tqdm(experiment_dirs):
            print(f'enter {d}')
            events = os.path.join(d, 'events.out*')
            events = command_result(f'ls {events}').split('\n')

            if len(events) != 1:
                print(f'{d}: error, found 5 {len(events)} files and skip.')
                continue

            event = events[0]
            print(f'processing {event}')

            event = event_accumulator.EventAccumulator(event)
            event.Reload()
            if 'Train/mean_reward' not in event.scalars.Keys():
                print(f'{d}: error, found 5 {len(events)} files and skip.')
            values = event.scalars.Items('Train/mean_reward')
            df = pd.DataFrame(values)
            df = df[df['step'] <= self._max_iterations]

            sava_path = os.path.join(self._save_dir, f'csv/{os.path.basename(d)}.csv')
            os.makedirs(os.path.dirname(sava_path), exist_ok=True)
            sava_path = os.path.abspath(sava_path)
            df.to_csv(sava_path)

            print(f'save {sava_path}')

    def table(self, table_type='none'):
        '''
        以表格的形式展示训练结果
        :param table_type: {none, latex}
        '''
        paper_table = []

        for asset in self._assets:
            algo_values = {}
            for algo in self._algos:
                dfs = []
                for file in glob.glob(os.path.join(self._save_dir, f"csv/bm_{asset}_{algo}*")):
                    df = pd.read_csv(file)
                    _, asset, algo, seed = os.path.basename(file.replace('.csv', '')).split("_")
                    dfs.append(df)

                if len(dfs) == 0:
                    continue

                dfs = pd.concat(dfs)
                mean_value = dfs[dfs['step'] > 2900]['value'].mean()
                std_value = dfs[dfs['step'] > 2900]['value'].std()

                algo_values[algo.upper()] = f'{mean_value:.2f}±{std_value:.2f}'

            paper_table.append(algo_values | {'asset': asset})
        df = pd.DataFrame(paper_table)
        df.reset_index(drop=True)
        df = df[['asset'] + list(map(str.upper, self._algos.keys()))]

        if table_type == 'none':
            print(df)
        elif table_type == 'latex':
            latex_text = df.to_latex(index=False)
            print(latex_text)




    def plot(self):
        """
        plot CSV data
        :return:
        """

        for asset in self._assets:
            self.plot_asset(asset)

    def plot_asset(self, asset):
        """
        plot CSV data
        :return:
        """
        colors = {
            'sac': 'green',
            'ppo': 'blue',
            'trpo': 'yellow',
            'td3': 'purple',
            'ddpg': 'red',
        }
        plt.rcParams['font.size'] = 18
        plt.cla()
        fig = plt.figure(figsize=(8, 6), dpi=500)
        for algo in self._algos:
            print(f'processing {algo}')
            datas = []
            for file in glob.glob(os.path.join(self._save_dir, f"csv/bm_{asset}_{algo}*")):
                print(f'load file: {file}')
                df = pd.read_csv(file)
                _, asset, algo, seed = os.path.basename(file.replace('.csv', '')).split("_")

                df['seed'] = seed
                df[df['value'] < -100] = -100
    
                w_size = 100
                df['value'] = np.convolve(df['value'], np.ones(w_size) / float(w_size), 'same')

                df = df[df['step'] % 20 == 0]
                df = df[df['step'] < self._max_iterations - w_size]
                datas.append(df)

            if len(datas) == 0:
                continue

            print(f'start polt...')
            datas = pd.concat(datas)
            sns.lineplot(datas, x='step', y='value', label=str.upper(algo), color=colors[algo])
            plt.legend()
            # plt.ylim([-75, 45])
            plt.xlabel('Episodes')
            labelpads = {
                'sup': -3,
                'wave': -3,
                'mixed': -3,
            }
            plt.ylabel('Average Reward', labelpad=labelpads.get(asset, None))
            plt.xlim([0, 3000])

        ax = plt.gca()
        def sc(x, pos):
            return "${:.1f}$".format(x/1e3, int(pos))

        # ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True, useOffset=True))
        ax.xaxis.set_major_formatter(FuncFormatter(sc))
        ax.annotate('1e3', xy=(0.96, -0.085), xycoords='axes fraction', ha='left', va='top')


        image_path = os.path.join(self._save_dir, f'{asset}')
        plt.savefig(image_path)
        print(f'save {image_path}')


if __name__ == '__main__':
    fire.Fire(Benchmark)


