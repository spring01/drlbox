
from drlbox.net import NoisyACNet
from .ac_eval import ACEvaluator


class NoisyNetACEvaluator(ACEvaluator):
    net_cls = NoisyACNet
