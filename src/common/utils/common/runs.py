import os
import glob


def get_all_runs_names():
    return [i for i in os.listdir('runs') if os.path.isdir(f'runs/{i}')]


def get_sarl_checkpoint_names(runs_name):
    return [i for i in os.listdir(f'runs/{runs_name}') if not i.startswith('events.out')]


def get_sarl_runs_dir():
    return ['/'.join(i.split('/')[:-1]) for i in glob.glob('runs/*/events.out*')]

def get_sarl_runs_names():
    return [i.split('/')[1] for i in glob.glob('runs/*/events.out*')]

def get_sarl_checkpoint_abspaths(runs_name):
    return [f'runs/{runs_name}/{i}' for i in get_sarl_checkpoint_names(runs_name)]


def get_sarl_checkpoint_abspath(runs_name, checkpoint_name):
    return os.path.abspath(f'runs/{runs_name}/{checkpoint_name}')
