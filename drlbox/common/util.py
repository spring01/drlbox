
import sys
import gym.spaces
import numpy as np


def discrete_action(action_space):
    return type(action_space) is gym.spaces.discrete.Discrete

def continuous_action(action_space):
    return type(action_space) is gym.spaces.box.Box

def softmax(logits, axis=-1):
    max_logits = logits.max(axis=axis, keepdims=True)
    sumexp_shifted = np.exp(logits - max_logits).sum(axis=axis, keepdims=True)
    logsumexp = max_logits + np.log(sumexp_shifted)
    probs = np.exp(logits - logsumexp)
    probs /= probs.sum(axis=axis, keepdims=True)
    return probs

def softmax_with_minprob(logits, minprob, axis=-1):
    probs = softmax(logits, axis=axis)
    return np.maximum(minprob, probs - minprob / logits.shape[1])


