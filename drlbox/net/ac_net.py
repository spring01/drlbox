"""Actor critic net"""
import tensorflow as tf
from drlbox.common.namescope import TF_NAMESCOPE
from drlbox.net.net_base import RLNet


class ACNet(RLNet):
    """Class for actor critic net"""

    LOGPI = 1.1447298858494002  # constant of log(pi)

    def set_model(self, model):
        """Set Keras model"""
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, tf_value = model.outputs
        self.tf_value = tf_value[:, 0]  # shape [batch_size, 1] -> [batch_size]

    def set_loss(self, entropy_weight=0.01, min_var=None, policy_type=None):
        """Set the loss function to be minimized
        Args:
            entropy_weight: float, weight of the entropy regularization term.
            min_var: float, only used in continuous action space. When the
                action is parameterized as a Gaussian, it is the lower bound
                of the Gaussian variance.
            policy_type: string, 'softmax' or 'gaussian'.
        """
        with tf.name_scope(TF_NAMESCOPE):
            tf_logits = self.tf_logits
            ph_advantage = tf.placeholder(tf.float32, [None])
            ph_target = tf.placeholder(tf.float32, [None])

            if policy_type == 'softmax':
                kfac_policy_loss = 'categorical_predictive', (tf_logits,)
                ph_action = tf.placeholder(tf.int32, [None])
                log_probs = tf.nn.log_softmax(tf_logits)
                onehot_act = tf.one_hot(ph_action, depth=tf_logits.shape[1])
                log_probs_act = tf.reduce_sum(log_probs * onehot_act, axis=1)
                if entropy_weight:
                    probs = tf.nn.softmax(tf_logits)
                    neg_entropy = tf.reduce_sum(probs * log_probs, axis=1)
            elif policy_type == 'gaussian':
                assert min_var is not None
                dim_action = tf_logits.shape[1] - 1
                ph_action = tf.placeholder(tf.float32, [None, dim_action])
                self.tf_mean = tf_logits[:, :-1]
                tf_var = tf.nn.softplus(tf_logits[:, -1])
                self.tf_var = tf.maximum(tf_var, min_var)
                kfac_policy_loss = ('normal_predictive',
                                    (self.tf_mean, self.tf_var))
                two_var = 2.0 * self.tf_var
                act_minus_mean = ph_action - self.tf_mean
                log_norm = tf.reduce_sum(act_minus_mean**2, axis=1) / two_var
                log_2pv = self.LOGPI + tf.log(two_var)
                log_probs_act = -(log_norm + 0.5 * int(dim_action) * log_2pv)
                if entropy_weight:
                    neg_entropy = 0.5 * (log_2pv + 1.0)
            else:
                raise ValueError('policy_type {} invalid'.format(policy_type))

            # loss
            policy_loss = -(log_probs_act * ph_advantage)
            value_loss = tf.squared_difference(ph_target, self.tf_value)
            self.tf_loss = policy_loss + value_loss
            if entropy_weight:
                self.tf_loss += neg_entropy * entropy_weight

            # error for prioritization: critic abs td error
            self.tf_error = tf.abs(ph_target - self.tf_value)

        # kfac loss register
        kfac_value_loss = 'normal_predictive', (self.tf_value,)
        self.kfac_loss_list = [kfac_policy_loss, kfac_value_loss]

        # placeholders
        self.ph_train_list = [self.ph_state, ph_action,
                              ph_advantage, ph_target]

    def action_values(self, state):
        """Return the output of the policy network, i.e., 'logits'."""
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def state_value(self, state):
        """Return the output of the value network."""
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def ac_values(self, state):
        """Return both outputs of the policy and value networks."""
        return self.sess.run([self.tf_logits, self.tf_value],
                             feed_dict={self.ph_state: state})


