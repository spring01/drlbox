
from drlbox.net import QNet
from drlbox.common.policy import EpsGreedy
from drlbox.common.util import discrete_action
from .eval_base import Evaluator


class DQNEvaluator(Evaluator):

    KEYWORD_DICT = {**Evaluator.KEYWORD_DICT,
                    **dict(policy_eps=0.01,)}
    net_cls = QNet

    def setup_algorithm(self, action_space):
        if not discrete_action(action_space):
            raise TypeError('DQN supports only discrete action.')
        self.policy = EpsGreedy(self.policy_eps)

