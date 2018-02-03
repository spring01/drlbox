
from .eval_base import Evaluator
from drlbox.net import ACNet
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import StochasticDisc, StochasticCont


class ACEvaluator(Evaluator):

    KEYWORD_DICT = {**Evaluator.KEYWORD_DICT,
                    **dict(policy_sto_cont_min_var=1e-4,)}
    net_cls = ACNet

    def setup_algorithm(self, action_space):
        if discrete_action(action_space):
            self.policy = StochasticDisc()
        elif continuous_action(action_space):
            self.policy = StochasticCont(low=action_space.low,
                                    high=action_space.high,
                                    min_var=self.policy_sto_cont_min_var)
        else:
            raise TypeError('Type of action_space not valid')

