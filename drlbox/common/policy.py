"""Policies for discrete/continuous action spaces"""
import numpy as np
from drlbox.common.util import softmax


class Policy:
    """Base policy class"""

    def select_action(self, *args, **kwargs):
        """Base class select_action method"""
        raise NotImplementedError('This method should be overriden.')


class RandomPolicy(Policy):
    """Random policy for discrete action space"""

    def __init__(self, num_act):
        """
        Args:
            num_act: number of the discrete actions
        """
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self):
        """Returns an integer representing the discrete action
        """
        return np.random.randint(0, self.num_act)



class EpsGreedyPolicy(Policy):
    """Epsilon greedy for discrete action space
    With probability epsilon select a random action; otherwise greedy
    (so that when epsilon = 0.0 it falls back to greedy policy).
    Works only with discrete actions
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, action_values):
        """Returns an integer representing the discrete action
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, action_values.size)
        else:
            return action_values.argmax()


class DecayEpsGreedyPolicy(EpsGreedyPolicy):
    """Epsilon greedy with linear decay, works only with discrete actions
    """

    def __init__(self, eps_start, eps_end, eps_delta):
        """
        Args:
            eps_start: float, starting value of epsilon
            eps_end: float, ending value of epsilon
            eps_delta: float, for every 'select_action' call, epsilon will be
                decreased by this quantity
        """
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_delta = eps_delta

    def select_action(self, action_values):
        """Returns an integer representing the discrete action
        """
        if np.random.rand() < self.epsilon:
            action = np.random.randint(0, action_values.size)
        else:
            action = action_values.argmax()
        self.epsilon = max(self.eps_end, self.epsilon - self.eps_delta)
        return action


class SoftmaxPolicy(Policy):
    """Softmax policy for discrete action space"""

    def select_action(self, action_values):
        """Returns an integer representing the discrete action
        'action_values' is supposed to be 'logits', i.e., before softmax
        """
        probs = softmax(action_values)
        return np.random.choice(len(probs), p=probs)


class GaussianPolicy(Policy):
    """Gaussian policy for continuous action space"""

    min_var=1e-4

    def __init__(self, low, high):
        """
        Args:
            low: np.array, each entry is the lower bound on that dimension
            high: np.array, each entry is the higher bound on that dimension
        """
        self.low = low
        self.high = high

    def select_action(self, action_values):
        """Returns an np.array of floats representing the continuous action
        'action_values' is supposed to be 'logits'. In continuous control,
        'action_values' is interpreted as a spherical Gaussian signal where
        action_values[:-1] is the mean, and action_values[-1] is the variance.
        """
        dim_action = len(action_values) - 1
        mean, var_param = action_values[:-1], action_values[-1]
        var = max(np.logaddexp(var_param, 0.0), self.min_var) # softplus
        action = np.random.multivariate_normal(mean, var * np.eye(dim_action))
        return np.clip(action, a_min=self.low, a_max=self.high)


