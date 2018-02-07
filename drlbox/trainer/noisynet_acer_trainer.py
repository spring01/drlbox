
from drlbox.net import NoisyACERNet
from .acer_trainer import ACERTrainer


class NoisyNetACERTrainer(ACERTrainer):
    KEYWORD_DICT = {**ACERTrainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=0.0,)}
    net_cls = NoisyACERNet

