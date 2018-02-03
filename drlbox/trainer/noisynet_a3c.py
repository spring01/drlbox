
from .a3c import A3CTrainer
from drlbox.net import NoisyACNet


class NoisyNetA3CTrainer(A3CTrainer):
    KEYWORD_DICT = {**A3CTrainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=0.0,)}
    net_cls = NoisyACNet

