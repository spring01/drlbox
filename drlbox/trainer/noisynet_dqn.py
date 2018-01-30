
from .dqn import DQNTrainer
from drlbox.net import NoisyQNet


class NoisyNetDQNTrainer(DQNTrainer):

    def setup_algorithm(self, action_space):
        super().setup_algorithm(action_space)
        self.net_cls = NoisyQNet

