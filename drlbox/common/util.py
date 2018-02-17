
import sys
import gym.spaces
import numpy as np


def discrete_action(action_space):
    return type(action_space) is gym.spaces.discrete.Discrete

def continuous_action(action_space):
    return type(action_space) is gym.spaces.box.Box

WARN_ARG_NOT_USED = 'Warning: argument "{}" set but not used'
def set_args(obj, default_kwargs, kwargs):
    # combine arguments from default_kwargs and kwargs
    all_kwargs = {}
    for keyword, value in default_kwargs.items():
        all_kwargs[keyword] = value
    # replace with user-specified arguments
    for keyword, value in kwargs.items():
        if keyword not in default_kwargs:
            print(WARN_ARG_NOT_USED.format(keyword), flush=True)
        all_kwargs[keyword] = value

    # set arguments
    for keyword, value in all_kwargs.items():
        setattr(obj, keyword, value)

    # print arguments
    print('#### All arguments ####', flush=True)
    for keyword, value in sorted(all_kwargs.items()):
        print('    {} = {}'.format(keyword, value), flush=True)

def softmax(logits, axis=None):
    max_logits = logits.max(axis=axis, keepdims=True)
    sumexp_shifted = np.exp(logits - max_logits).sum(axis=axis, keepdims=True)
    logsumexp = max_logits + np.log(sumexp_shifted)
    probs = np.exp(logits - logsumexp)
    probs /= probs.sum(axis=axis, keepdims=True)
    return probs

