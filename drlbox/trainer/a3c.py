
from .trainer_base import Trainer
from drlbox.net import ACNet
from drlbox.rollout import RolloutAC
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import StochasticDisc, StochasticCont


class A3CTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=1e-2,
                           policy_sto_cont_min_var=1e-4,)}
    net_cls = ACNet

    def setup_algorithm(self, action_space):
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                min_var=self.policy_sto_cont_min_var)
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)
        self.rollout_builder = lambda s: RolloutAC(s, self.discount)
        if discrete_action(action_space):
            self.policy = StochasticDisc()
        elif continuous_action(action_space):
            self.policy = StochasticCont(low=action_space.low,
                                    high=action_space.high,
                                    min_var=self.policy_sto_cont_min_var)
        else:
            raise TypeError('Type of action_space not valid')

