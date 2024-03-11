import gym
from gym import spaces

def get_actions_num(action_space):
    if isinstance(action_space, spaces.Discrete):
        actions_num = action_space.n
    elif isinstance(action_space, spaces.Tuple):
        actions_num = [i.n for i in action_space]
    elif isinstance(action_space, spaces.Box):
        actions_num = action_space.shape[0]
    else:
        raise NotImplementedError()

    return actions_num

def get_input_shape(observation_space):
    if isinstance(observation_space, spaces.Box):
        input_shape = observation_space.shape[0]
    else:
        raise NotImplementedError()

    return input_shape