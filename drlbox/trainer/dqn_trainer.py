
import tensorflow as tf
from drlbox.net import QNet
from drlbox.common.policy import DecayEpsGreedy
from drlbox.common.util import discrete_action
from .trainer_base import Trainer


class DQNTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(policy_eps_start=1.0,
                           policy_eps_end=0.01,
                           policy_eps_decay_steps=1000000,
                           interval_sync_target=1000,)}
    net_cls = QNet

    def setup_algorithm(self, action_space):
        self.loss_kwargs = {}
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)

        # policy
        if not discrete_action(action_space):
            raise TypeError('DQN supports only discrete action.')
        eps_start = self.policy_eps_start
        eps_end = self.policy_eps_end
        eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
        self.policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)

    def setup_nets(self, worker_dev, rep_dev, env):
        super().setup_nets(worker_dev, rep_dev, env)
        with tf.device(worker_dev):
            self.target_net = self.build_net(env)
            self.target_net.set_sync_weights(self.global_net.weights)
        self.batch_counter = 0

    def set_session(self, sess):
        super().set_session(sess)
        self.target_net.set_session(sess)
        self.target_net.sync()

    def train_on_rollout_list(self, rollout_list):
        batch_loss = super().train_on_rollout_list(rollout_list)
        self.batch_counter += 1
        if self.batch_counter > self.interval_sync_target:
            self.batch_counter = 0
            self.target_net.sync()
        return batch_loss

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = self.rollout_state_input_action(rollout)
        last_state = r_state[-1:]
        online_last_value = self.online_net.action_values(last_state)[-1]
        target_last_value = self.target_net.action_values(last_state)[-1]
        target_last_q = target_last_value[online_last_value.argmax()]
        r_target = self.rollout_target(rollout, target_last_q)
        return r_input, r_action, r_target

