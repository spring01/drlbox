
from drlbox.net import ACNet
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import SoftmaxPolicy, GaussianPolicy
from .trainer_base import Trainer


class A3CTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=1e-2,
                           policy_sto_cont_min_var=1e-4,)}
    net_cls = ACNet

    def setup_algorithm(self, action_space):
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                min_var=self.policy_sto_cont_min_var)
        if discrete_action(action_space):
            self.policy = SoftmaxPolicy()
        elif continuous_action(action_space):
            self.policy = GaussianPolicy(low=action_space.low,
                                         high=action_space.high,
                                         min_var=self.policy_sto_cont_min_var)
        else:
            raise TypeError('Invalid type of action_space')

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = self.rollout_state_input_action(rollout)
        r_value = self.online_net.state_value(r_state)
        r_target = self.rollout_target(rollout, r_value[-1])
        r_adv = r_target - r_value[:-1]
        return r_input, r_action, r_adv, r_target

