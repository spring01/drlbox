
from .dqn_eval import DQNEvaluator
from drlbox.net import NoisyQNet


class NoisyNetDQNEvaluator(DQNEvaluator):
    net_cls = NoisyQNet

