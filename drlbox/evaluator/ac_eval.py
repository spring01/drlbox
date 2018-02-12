
from drlbox.net import ACNet
from .eval_base import Evaluator


class ACEvaluator(Evaluator):
    net_cls = ACNet

