
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
            print(WARN_ARG_NOT_USED.format(keyword), flush=True)
        setattr(obj, keyword, value)

def softmax(logits, axis=None):
    max_logits = logits.max(axis=axis, keepdims=True)
    sumexp_shifted = np.exp(logits - max_logits).sum(axis=axis, keepdims=True)
    logsumexp = max_logits + np.log(sumexp_shifted)
    probs = np.exp(logits - logsumexp)
    probs /= probs.sum(axis=axis, keepdims=True)
    return probs

