
import numpy as np


class Policy(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--policy_eps', default=0.01, type=float,
            help='Exploration probability in epsilon-greedy')
        parser.add_argument('--policy_decay_from', default=1.0, type=float,
            help='Starting probability in linear-decay epsilon-greedy')
        parser.add_argument('--policy_decay_to', default=0.1, type=float,
            help='Ending probability in linear-decay epsilon-greedy')
        parser.add_argument('--policy_decay_steps', default=2000000, type=int,
            help='Decay steps in linear-decay epsilon-greedy')

    def select_action(self, *args, **kwargs):
        raise NotImplementedError('This method should be overriden.')


class RandomPolicy(Policy):

    def __init__(self, num_act):
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self, *args):
        return np.random.randint(0, self.num_act)


class GreedyEpsPolicy(Policy):

    def __init__(self, args):
        self.epsilon = args.policy_eps

    def select_action(self, q_values, *args):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)


class LinearDecayGreedyEpsPolicy(Policy):

    def __init__(self, args):
        self.start_value = args.policy_decay_from
        self.end_value = args.policy_decay_to
        self.num_steps = float(args.policy_decay_steps)

    def select_action(self, q_values, iter_num=0):
        wt_end = min(iter_num / self.num_steps, 1.0)
        wt_start = 1.0 - wt_end
        epsilon = self.start_value * wt_start + self.end_value * wt_end
        if np.random.rand() <= epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)
