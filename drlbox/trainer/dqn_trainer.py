
import numpy as np
import tensorflow as tf
from drlbox.net import QNet
from drlbox.common.policy import DecayEpsGreedyPolicy
from drlbox.common.util import discrete_action
from .trainer_base import Trainer


class DQNTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(dqn_double=True,
                           policy_eps_start=1.0,
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
        concat_len = cc_state.shape[0]
        cc_target_value = self.concat_value(self.target_net, concat_len,
                                            last_states, rl_slice)
        if self.dqn_double:
            cc_online_value = self.concat_value(self.online_net, concat_len,
                                                last_states, rl_slice)
            return cc_target_value, cc_online_value
        else:
            return cc_target_value,

    def concat_value(self, net, concat_len, last_states, rl_slice):
        last_values = net.action_values(last_states)
        value_shape = concat_len, *last_values.shape[1:]
        cc_value = np.zeros(value_shape)
        for r_slice, last_val in zip(rl_slice, last_values):
            cc_value[r_slice][-1] = last_val
        return cc_value

    def rollout_feed(self, rollout, r_target_value, r_online_value=None):
        r_action = np.array(rollout.action_list)
        if self.dqn_double:
            greedy_last_value = r_online_value[-1]
        else:
            greedy_last_value = r_target_value[-1]
        target_last_q = r_target_value[-1][greedy_last_value.argmax()]
        r_target = self.rollout_target(rollout, target_last_q)
        return r_action, r_target

