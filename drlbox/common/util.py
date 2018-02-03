
import sys
import gym.spaces


def discrete_action(action_space):
    return type(action_space) is gym.spaces.discrete.Discrete

def continuous_action(action_space):
    return type(action_space) is gym.spaces.box.Box

WARN_ARG_NOT_USED = 'Warning: argument "{}" set but not used'
def set_args(obj, default_kwargs, kwargs):
    # set default arguments
    for keyword, value in default_kwargs.items():
        setattr(obj, keyword, value)
    # replace with user-specified arguments
    for keyword, value in kwargs.items():
        if keyword not in default_kwargs:
            print(WARN_ARG_NOT_USED.format(keyword), file=sys.stderr)
        setattr(obj, keyword, value)

