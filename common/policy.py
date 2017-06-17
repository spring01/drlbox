
import numpy as np


''' Works with discrete actions '''
class Policy(object):

    def select_action(self, *args, **kwargs):
        raise NotImplementedError('This method should be overriden.')


class Random(Policy):

    def __init__(self, num_act):
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self):
        return np.random.randint(0, self.num_act)


class EpsGreedy(Policy):

    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)


class LinearDecayEpsGreedy(EpsGreedy):

    def __init__(self, start_eps, end_eps, decay_steps):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.decay_steps = float(decay_steps)
        self.update(0)

    def update(self, step):
        wt_end = min(step / self.decay_steps, 1.0)
        wt_start = 1.0 - wt_end
        self.epsilon = self.start_eps * wt_start + self.end_eps * wt_end


class Stochastic(Policy):

    def __init__(self, num_act):
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self, probs):
        return np.random.choice(self.num_act, p=probs)

