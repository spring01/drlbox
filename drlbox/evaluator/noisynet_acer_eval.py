
from .acer_eval import ACEREvaluator
from drlbox.net import NoisyACERNet


class NoisyNetACEREvaluator(ACEREvaluator):
    net_cls = NoisyACERNet
