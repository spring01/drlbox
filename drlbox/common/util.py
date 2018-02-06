
import sys
import gym.spaces
import numpy as np


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

def softmax(logits):
    max_value = logits.max()
    sumexp_shifted = np.sum(np.exp(logits - max_value))
    logsumexp = max_value + np.log(sumexp_shifted)
    probs = np.exp(logits - logsumexp)
    probs /= np.sum(probs)
    return probs

