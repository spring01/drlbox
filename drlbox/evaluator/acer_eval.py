
from .ac_eval import ACEvaluator
from drlbox.net import ACERNet


class ACEREvaluator(ACEvaluator):
    net_cls = ACERNet

