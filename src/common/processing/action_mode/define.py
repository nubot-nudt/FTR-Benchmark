from abc import abstractmethod
from gym import spaces

class ActionModeFactory:
    _registered_action_mode = {}

    @classmethod
    def register(cls, name, cls_name):
        if name not in cls._registered_action_mode:
            cls._registered_action_mode[name] = cls_name
        else:
            raise KeyError(f'action_mode {name} already exists')

    @classmethod
    def get_action_mode_by_name(cls, name, *args, **kwargs):
        return cls._registered_action_mode[name](*args, **kwargs)

    @classmethod
    def get_names(cls):
        return list(cls._registered_action_mode.keys())

def get_action_mode_gym_space(action_mode):
    return ActionModeFactory.get_action_mode_by_name(action_mode).gym_space


class ActionMode:

    def __init__(self, max_v=0.5, flipper_dt=2, max_w=None):
        self.max_v = max_v
        self.max_w = self.max_v if max_w is None else max_w
        self.flipper_dt = flipper_dt

    @property
    def gym_info(self):

        if isinstance(self.gym_space, spaces.Discrete):
            num_actions = 1
        elif isinstance(self.gym_space, spaces.Tuple):
            num_actions = len(self.gym_space)
        elif isinstance(self.gym_space, spaces.Box):
            num_actions = self.gym_space.shape[0]
        else:
            raise NotImplementedError()

        return {
            'num_actions': num_actions,
            'space': self.gym_space,
        }

    @property
    def gym_space(self):
        pass

    @abstractmethod
    def convert_actions_to_std_dict(self, actions, default_v=0.2, default_w=0.0):
        pass
