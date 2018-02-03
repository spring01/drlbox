
from .ac_eval import ACEvaluator
from drlbox.net import NoisyACNet


class NoisyNetACEvaluator(ACEvaluator):
    net_cls = NoisyACNet
