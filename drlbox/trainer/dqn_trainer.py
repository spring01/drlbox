"""DQN trainer"""
import numpy as np
import tensorflow as tf
from drlbox.net import QNet
from drlbox.common.policy import DecayEpsGreedyPolicy
from drlbox.trainer.trainer_base import Trainer


DQN_KWARGS = dict(
    dqn_double=True,
    dqn_dueling=False,
    policy_eps_start=1.0,
    policy_eps_end=0.01,
    policy_eps_decay_steps=1000000,
    sync_target_interval=1000,
    )

class DQNTrainer(Trainer):
    """DQN trainer"""

    KWARGS = {**Trainer.KWARGS, **DQN_KWARGS}
    net_cls = QNet

    def setup_algorithm(self):
        """Setup properties required by DQN."""
        self.loss_kwargs = {}
        assert self.action_mode == 'discrete'

        # epsilon greedy policy
        eps_start = self.policy_eps_start
        eps_end = self.policy_eps_end
        eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
        self.policy = DecayEpsGreedyPolicy(eps_start, eps_end, eps_delta)
        self.model_kwargs = {}

    def setup_nets(self, worker_dev, rep_dev, env):
        """Setup the target network in DQN."""
        super().setup_nets(worker_dev, rep_dev, env)
        with tf.device(worker_dev):
            self.target_net = self.build_net(env)
            self.target_net.set_sync_weights(self.global_net.weights)
        self.batch_counter = 0

    def build_model(self, state, feature):
        """Return a Keras model."""
        assert self.action_mode == 'discrete'
        if self.dqn_dueling:
            if type(feature) is tuple:
                assert len(feature) == 2
                # separated adv/value streams when feature is a length 2 tuple
                feature_adv, feature_value = feature
            else:
                size_last_hid = feature.shape.as_list()[-1]
                assert size_last_hid % 2 == 0
                size_dueling = size_last_hid // 2
                # split feature in halves
                l_first_half = lambda x: x[..., :size_dueling]
                l_last_half = lambda x: x[..., size_dueling:]
                feature_adv = tf.keras.layers.Lambda(l_first_half)(feature)
                feature_value = tf.keras.layers.Lambda(l_last_half)(feature)
            hid_adv = tf.keras.layers.Dense(size_dueling)(feature_adv)
            hid_adv = tf.keras.layers.Activation('relu')(hid_adv)
            hid_value = tf.keras.layers.Dense(size_dueling)(feature_value)
            hid_value = tf.keras.layers.Activation('relu')(hid_value)
            adv = self.dense_layer(self.action_dim)(hid_adv)
            value = self.dense_layer(1)(hid_value)
            l_baseline = lambda x: -tf.reduce_mean(x, axis=-1, keepdims=True)
            baseline = tf.keras.layers.Lambda(l_baseline)(adv)
            q_value = tf.keras.layers.Add()([value, baseline, adv])
        else:
            q_value = self.dense_layer(self.action_dim)(feature)
        model = tf.keras.models.Model(inputs=state, outputs=q_value)
        return model

    def set_session(self, sess):
        """Setup a TensorFlow session."""
        super().set_session(sess)
        self.target_net.set_session(sess)
        self.target_net.sync()

    def sync_to_global(self):
        """Sync the local network to the global network and resample weight
        noises in the target network if necessary.
        """
        super().sync_to_global()
        if self.noisynet is not None:
            self.target_net.sample_noise()

    def train_on_batch(self, *args):
        """Train on a batch."""
        batch_result = super().train_on_batch(*args)
        self.batch_counter += 1
        if self.batch_counter >= self.sync_target_interval:
            self.batch_counter = 0
            self.target_net.sync()
        return batch_result

    def concat_bootstrap(self, cc_state, b_r_slice):
        """Return concatenated bootstrap training target values."""
        last_states = [cc_state[r_slice][-1] for r_slice in b_r_slice]
        last_states = np.array(last_states)
        concat_len = cc_state.shape[0]
        cc_target_value = concat_value(self.target_net, concat_len,
                                       last_states, b_r_slice)
        if self.dqn_double:
            cc_online_value = concat_value(self.online_net, concat_len,
                                           last_states, b_r_slice)
            return cc_target_value, cc_online_value
        else:
            return cc_target_value,

    def rollout_feed(self, rollout, r_target_value, r_online_value=None):
        """Return training targets for a rollout."""
        r_action = np.array(rollout.action_list)
        if self.dqn_double:
            greedy_last_value = r_online_value[-1]
        else:
            greedy_last_value = r_target_value[-1]
        target_last_q = r_target_value[-1][greedy_last_value.argmax()]
        r_target = self.rollout_target(rollout, target_last_q)
        return r_action, r_target


def concat_value(net, concat_len, last_states, b_r_slice):
    """Return one type of concatenated value."""
    last_values = net.action_values(last_states)
    value_shape = concat_len, *last_values.shape[1:]
    cc_value = np.zeros(value_shape)
    for r_slice, last_val in zip(b_r_slice, last_values):
        cc_value[r_slice][-1] = last_val
    return cc_value
