
from drlbox.net import NoisyQNet
from .dqn_trainer import DQNTrainer


class NoisyNetDQNTrainer(DQNTrainer):
    net_cls = NoisyQNet

