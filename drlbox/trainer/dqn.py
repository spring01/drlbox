
from .base import Trainer
from drlbox.net import QNet
from drlbox.async.rollout import RolloutMultiStepQ
from drlbox.common.policy import DecayEpsGreedy


class DQNTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(policy_eps_start=1.0,
                           policy_eps_end=0.01,
                           policy_eps_decay_steps=1000000,)}

    need_target_net = True

    def setup_algorithm(self, action_space):
        self.net_cls = QNet
        self.loss_kwargs = {}
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)

        # rollout
        self.rollout_builder = lambda s: RolloutMultiStepQ(s, self.discount)

        # policy
        eps_start = self.policy_eps_start
        eps_end = self.policy_eps_end
        eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
        self.policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)

