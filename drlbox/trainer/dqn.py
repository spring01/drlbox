
from .trainer_base import Trainer
from drlbox.net import QNet
from drlbox.rollout import RolloutMultiStepQ
from drlbox.common.policy import DecayEpsGreedy
from drlbox.common.util import discrete_action


class DQNTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(policy_eps_start=1.0,
                           policy_eps_end=0.01,
                           policy_eps_decay_steps=1000000,)}
    net_cls = QNet
    need_target_net = True

    def setup_algorithm(self, action_space):
        self.loss_kwargs = {}
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)

        # rollout
        self.rollout_builder = lambda s: RolloutMultiStepQ(s, self.discount)

        # policy
        if not discrete_action(action_space):
            raise TypeError('DQN supports only discrete action.')
        eps_start = self.policy_eps_start
        eps_end = self.policy_eps_end
        eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
        self.policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)

