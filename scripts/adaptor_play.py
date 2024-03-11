
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src/common'))
from utils.path import apply_project_directory
apply_project_directory()

from isaacgym_ext.start import launch_isaacgym_env
from isaacgym_ext.interface.adaptor import KeyboardEnvAdaptor
from isaacgym_ext.interface.keyboard import KeyboardEnv

if __name__ == '__main__':
    def preprocess_func(config):
        config['headless'] = False
        config['test'] = True
        config['task']['env']['actionMode'] = 'continuous_std_6'
        config['task']['env']['numEnvs'] = 1


    with launch_isaacgym_env(preprocess_func) as ret:
        cfg_dict = ret['config']
        env = ret['env']

        env = KeyboardEnv(KeyboardEnvAdaptor(env))
        env.run()



