
from drlbox.net import QNet
from .eval_base import Evaluator


class DQNEvaluator(Evaluator):
    net_cls = QNet

