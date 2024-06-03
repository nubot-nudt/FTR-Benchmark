
import base64
import pickle


from ptask_envs.envs.start import load_hydra_config

if __name__ == '__main__':
    cfg = load_hydra_config(convert_dict=True)
    print(base64.b64encode(pickle.dumps(cfg)))
