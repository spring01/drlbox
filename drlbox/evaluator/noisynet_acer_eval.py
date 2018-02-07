
from drlbox.net import NoisyACERNet
from .acer_eval import ACEREvaluator


class NoisyNetACEREvaluator(ACEREvaluator):
    net_cls = NoisyACERNet
