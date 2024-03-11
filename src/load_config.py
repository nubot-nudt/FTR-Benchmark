
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), 'common'))
from utils.path import apply_project_directory
apply_project_directory()

import base64
import pickle


from isaacgym_ext.start import load_hydra_config

if __name__ == '__main__':
    cfg = load_hydra_config(convert_dict=True)
    print(base64.b64encode(pickle.dumps(cfg)))
