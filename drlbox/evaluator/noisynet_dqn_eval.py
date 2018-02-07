
from drlbox.net import NoisyQNet
from .dqn_eval import DQNEvaluator


class NoisyNetDQNEvaluator(DQNEvaluator):
    net_cls = NoisyQNet

