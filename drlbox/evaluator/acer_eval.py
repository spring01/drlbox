
from drlbox.net import ACERNet
from .ac_eval import ACEvaluator


class ACEREvaluator(ACEvaluator):
    net_cls = ACERNet

