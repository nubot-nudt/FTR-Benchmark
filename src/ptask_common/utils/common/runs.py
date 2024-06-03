import os
import glob
from pathlib import Path
from deprecated import deprecated


def get_all_runs_names():
    return [i for i in os.listdir('runs') if os.path.isdir(f'runs/{i}')]


def get_ftr_runs(logdir):
    runs = []
    directories = glob.glob(logdir) if '*' in logdir else os.listdir(logdir)
    logdir = Path(logdir)
    for directory in directories:
        if directory.startswith('bm_'):
            runs.append(str((logdir / directory).absolute()))
            continue

        child = os.listdir(logdir / directory)
        if any(map(lambda x: x.startswith('events.out'), child)):
            runs.append(str((logdir / directory).absolute()))
            continue

        if len(set(['mappo', 'hatrpo', 'hatrpo']) & set(child)) > 0:
            runs.append(str((logdir / directory).absolute()))
            continue
    return runs


@deprecated
def get_sarl_checkpoint_names(runs_name):
    return [i for i in os.listdir(f'runs/{runs_name}') if not i.startswith('events.out')]


@deprecated
def get_sarl_runs_dir(runs='runs'):
    return ['/'.join(i.split('/')[:-1]) for i in glob.glob(f'{runs}/*/events.out*')]


@deprecated
def get_sarl_runs_names():
    return [i.split('/')[1] for i in glob.glob('runs/*/events.out*')]


@deprecated
def get_sarl_checkpoint_abspaths(runs_name):
    return [f'runs/{runs_name}/{i}' for i in get_sarl_checkpoint_names(runs_name)]


@deprecated
def get_sarl_checkpoint_abspath(runs_name, checkpoint_name):
    return os.path.abspath(f'runs/{runs_name}/{checkpoint_name}')
