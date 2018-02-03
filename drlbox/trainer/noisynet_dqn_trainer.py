
from .dqn_trainer import DQNTrainer
from drlbox.net import NoisyQNet


class NoisyNetDQNTrainer(DQNTrainer):
    net_cls = NoisyQNet

