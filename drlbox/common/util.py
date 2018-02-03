
import gym.spaces


def discrete_action(action_space):
    return type(action_space) is gym.spaces.discrete.Discrete

def continuous_action(action_space):
    return type(action_space) is gym.spaces.box.Box

def set_args(obj, default_kwargs, kwargs):
    # set default arguments
    for keyword, value in default_kwargs.items():
        setattr(obj, keyword, value)
    # replace with user-specified arguments
    for keyword, value in kwargs.items():
        if keyword not in default_kwargs:
            raise ValueError('Argument "{}" not valid'.format(keyword))
        setattr(obj, keyword, value)

