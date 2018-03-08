
import numpy as np
import tensorflow as tf
from drlbox.net import QNet
from drlbox.common.policy import DecayEpsGreedyPolicy
from drlbox.common.util import discrete_action
from .trainer_base import Trainer


class DQNTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(policy_eps_start=1.0,
                           policy_eps_end=0.01,
                           policy_eps_decay_steps=1000000,
                           interval_sync_target=1000,
                           )}
    net_cls = QNet

    def setup_algorithm(self, action_space):
        self.loss_kwargs = {}

        # policy
        if not discrete_action(action_space):
            raise TypeError('action_space must be discrete in DQN')
        eps_start = self.policy_eps_start
        eps_end = self.policy_eps_end
        eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
        self.policy = DecayEpsGreedyPolicy(eps_start, eps_end, eps_delta)

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

    def train_on_batch(self, batch):
        batch_loss = super().train_on_batch(batch)
        self.batch_counter += 1
        if self.batch_counter >= self.interval_sync_target:
            self.batch_counter = 0
            self.target_net.sync()
        return batch_loss

    def concat_bootstrap(self, cc_state, rl_slice):
        last_states = [cc_state[r_slice][-1] for r_slice in rl_slice]
        last_states = np.array(last_states)
        online_last = self.online_net.action_values(last_states)
        target_last = self.target_net.action_values(last_states)
        value_shape = cc_state.shape[0], *online_last.shape[1:]
        cc_online_value = np.zeros(value_shape)
        cc_target_value = np.zeros(value_shape)
        for r_slice, o_last, t_last in zip(rl_slice, online_last, target_last):
            cc_online_value[r_slice][-1] = o_last
            cc_target_value[r_slice][-1] = t_last
        return cc_online_value, cc_target_value

    def rollout_feed(self, rollout, r_online_value, r_target_value):
        r_action = np.array(rollout.action_list)
        online_last_value = r_online_value[-1]
        target_last_value = r_target_value[-1]
        target_last_q = target_last_value[online_last_value.argmax()]
        r_target = self.rollout_target(rollout, target_last_q)
        return r_action, r_target

