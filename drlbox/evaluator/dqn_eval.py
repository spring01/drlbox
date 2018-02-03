
from .eval_base import Evaluator
from drlbox.net import QNet
from drlbox.common.policy import EpsGreedy


class DQNEvaluator(Evaluator):

    KEYWORD_DICT = {**Evaluator.KEYWORD_DICT,
                    **dict(policy_eps=0.01,)}
    net_cls = QNet

    def setup_algorithm(self, action_space):
        self.policy = EpsGreedy(self.policy_eps)

