
from drlbox.net import NoisyACNet
from .a3c_trainer import A3CTrainer


class NoisyNetA3CTrainer(A3CTrainer):
    KEYWORD_DICT = {**A3CTrainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=0.0,)}
    net_cls = NoisyACNet

