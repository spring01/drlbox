
import numpy as np


''' Works with discrete actions '''
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

    def update(self, *args, **kwargs):
        pass


class RandomPolicy(Policy):

    def __init__(self, num_act):
        assert num_act >= 1
        self.num_act = num_act

    def select_action(self, *args, **kwargs):
        return np.random.randint(0, self.num_act)


class GreedyEpsPolicy(Policy):

    def __init__(self, args):
        self.epsilon = args.policy_eps

    def select_action(self, q_values):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, q_values.size)
        else:
            return np.argmax(q_values)


class LinearDecayGreedyEpsPolicy(GreedyEpsPolicy):

    def __init__(self, args):
        self.start_eps = args.policy_decay_from
        self.end_eps = args.policy_decay_to
        self.num_steps = float(args.policy_decay_steps)

    def update(self, step_count):
        wt_end = min(step_count / self.num_steps, 1.0)
        wt_start = 1.0 - wt_end
        self.epsilon = self.start_eps * wt_start + self.end_eps * wt_end


