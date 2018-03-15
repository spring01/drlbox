
import numpy as np
from drlbox.common.util import softmax


class Policy:

    def select_action(self, *args, **kwargs):
        raise NotImplementedError('This method should be overriden.')


class RandomPolicy(Policy):

    def __init__(self, num_act):
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self):
        return np.random.randint(0, self.num_act)


'''
With probability epsilon select a random action; otherwise greedy
(so that when epsilon = 0.0 it falls back to greedy policy).
Works only with discrete actions
'''
class EpsGreedyPolicy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, action_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, action_values.size)
        else:
            return action_values.argmax()

'''
Also works only with discrete actions
'''
class DecayEpsGreedyPolicy(EpsGreedyPolicy):

    def __init__(self, eps_start, eps_end, eps_delta):
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_delta = eps_delta

    def select_action(self, action_values):
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, action_values.size)
        else:
            action = action_values.argmax()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_delta)
        return action


class SoftmaxPolicy(Policy):

    '''
    'action_values' is supposed to be 'logits', i.e., before softmax
    '''
    def select_action(self, action_values):
        probs = softmax(action_values)
        return np.random.choice(len(probs), p=probs)


class GaussianPolicy(Policy):

    min_var=1e-4

    def __init__(self, low, high):
        self.low = low
        self.high = high

    '''
    'action_values' is supposed to be 'logits'. In continuous control,
    'action_values' is interpreted as a spherical Gaussian signal where
    action_values[:-1] is the mean, and action_values[-1] is the variance.
    '''
    def select_action(self, action_values):
        dim_action = len(action_values) - 1
        mean, var_param = action_values[:-1], action_values[-1]
        var = max(np.logaddexp(var_param, 0.0), self.min_var) # softplus
        action = np.random.multivariate_normal(mean, var * np.eye(dim_action))
        return np.clip(action, a_min=self.low, a_max=self.high)


