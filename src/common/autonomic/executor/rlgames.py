import torch

from rlgames_ext.loader.common import ModelLoader
from rlgames_ext.loader.factory import LoaderFactory

from autonomic.common.base import BaseExecutor
from processing.robot_info.base import RobotState
from processing.action_mode.define import ActionModeFactory

class RlgamesPredictor(BaseExecutor):

    @classmethod
    def load_model(cls, runs_dir, epoch=-1, device='cpu', input_shape=None, actions_num=None, action_space=None, observation_space=None):

        return cls(LoaderFactory.get_loader_by_runs(runs_dir=runs_dir, epoch=epoch, device=device,
                                   input_shape=input_shape,
                                   actions_num=actions_num,
                                   action_space=action_space,
                                   observation_space=observation_space,))

    def __init__(self, model_loader: ModelLoader, is_deterministic=False):
        self.model_loader = model_loader

        self.is_deterministic = is_deterministic

        params = self.model_loader.params

        if 'space' in params:
            self.is_multi_discrete = 'multi_discrete' in params['space']
            self.is_discrete = 'discrete' in params['space']
            self.is_continuous = 'continuous' in params['space']
        else:
            self.is_discrete = False
            self.is_continuous = False
            self.is_multi_discrete = False

        self.action_mode = ActionModeFactory.get_action_mode_by_name(self.model_loader.cfg['task']['task']['actionMode'])


    @property
    def input_shape(self):
        return self.model_loader.input_shape

    def predict(self, state: RobotState):
        obs = torch.cat([state.height_map.flatten(), state.vels, torch.deg2rad(state.flippers), state.orient, torch.tensor([0.3, 0, 0])])

        return (self.get_action(obs.float()) - 1) * 2


    def get_action(self, obs):
        return self.model_loader.get_action(obs)

        # is_multi_discrete = self.is_multi_discrete
        # is_deterministic = self.is_deterministic
        #
        # input_dict = {
        #     'is_train': False,
        #     'is_play': True,
        #     'obs': obs.view(-1, self.input_shape),
        # }
        #
        # with torch.no_grad():
        #     res_dict = self.model_loader(input_dict)
        #
        # logits = res_dict['logits']
        # action = res_dict['actions']
        # self.states = res_dict['rnn_states']
        #
        # if is_multi_discrete:
        #     if is_deterministic:
        #         action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
        #         return torch.stack(action, dim=-1)
        #     else:
        #         return action.squeeze().detach()
        # else:
        #     if is_deterministic:
        #         return torch.argmax(logits.detach(), axis=-1).squeeze()
        #     else:
        #         return action.squeeze().detach()


    def get_value(self, obs):

        # input_dict = {
        #     'is_train': False,
        #     'is_play': True,
        #     'prev_actions': None,
        #     'obs': obs.view(-1, self.input_shape),
        # }
        # result = self.model_loader(input_dict)
        # value = result['values']
        # return value
        return self.model_loader.get_value(obs)