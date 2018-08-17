"""A3C trainer"""
import numpy as np
import tensorflow as tf
from drlbox.net import ACNet
from drlbox.common.policy import SoftmaxPolicy, GaussianPolicy
from drlbox.trainer.trainer_base import Trainer


A3C_KWARGS = dict(
    a3c_entropy_weight=1e-2,
    )

class A3CTrainer(Trainer):
    """A3C trainer"""

    KWARGS = {**Trainer.KWARGS, **A3C_KWARGS}
    net_cls = ACNet

    def setup_algorithm(self):
        """Setup properties required by A3C."""
        if self.action_mode == 'discrete':
            policy_type = 'softmax'
            size_logits = self.action_dim
            size_value = 1
            logits_init = tf.keras.initializers.RandomNormal(stddev=1e-3)
            self.policy = SoftmaxPolicy()
        elif self.action_mode == 'continuous':
            policy_type = 'gaussian'
            size_logits = self.action_dim + 1
            size_value = 1
            logits_init = 'glorot_uniform'
            self.policy = GaussianPolicy(low=self.action_low,
                                         high=self.action_high)
        else:
            raise TypeError('action_mode {} invalid'.format(self.action_mode))
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                min_var=GaussianPolicy.min_var,
                                policy_type=policy_type)
        self.model_kwargs = dict(size_logits=size_logits,
                                 size_value=size_value,
                                 logits_init=logits_init)

    def build_model(self, state, feature,
                    size_logits=None, size_value=None, logits_init=None):
        """Return a Keras model."""
        if type(feature) in {tuple, list}:
            assert len(feature) == 2
            # separated logits/value streams when feature is a length 2 tuple
            feature_logits, feature_value = feature
        else:
            # feature is a single stream otherwise
            feature_logits = feature_value = feature
        logits_layer = self.dense_layer(size_logits,
                                        kernel_initializer=logits_init)
        logits = logits_layer(feature_logits)
        value = self.dense_layer(size_value)(feature_value)
        model = tf.keras.models.Model(inputs=state, outputs=[logits, value])
        return model

    def concat_bootstrap(self, cc_state, b_r_slice):
        """Return bootstrapped quantities for a concatenated batch."""
        cc_value = self.online_net.state_value(cc_state)
        return cc_value, # should return a tuple

    def rollout_feed(self, rollout, r_value):
        """Return feeds for a rollout."""
        r_action = np.array(rollout.action_list)
        r_target = self.rollout_target(rollout, r_value[-1])
        r_adv = r_target - r_value[:-1]
        return r_action, r_adv, r_target
