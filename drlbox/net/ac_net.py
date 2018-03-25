
import tensorflow as tf
from .net_base import RLNet


class ACNet(RLNet):

    LOGPI = 1.1447298858494002

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, tf_value = model.outputs
        self.tf_value = tf_value[:, 0]

    def set_loss(self, entropy_weight=0.01, min_var=None, policy_type=None):
        tf_logits = self.tf_logits
        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])

        if policy_type == 'softmax':
            kfac_policy_loss = 'categorical_predictive', (tf_logits,)
            ph_action = tf.placeholder(tf.int32, [None])
            log_probs = tf.nn.log_softmax(tf_logits)
            action_onehot = tf.one_hot(ph_action, depth=tf_logits.shape[1])
            log_probs_act = tf.reduce_sum(log_probs * action_onehot, axis=1)
            if entropy_weight:
                probs = tf.nn.softmax(tf_logits)
                neg_entropy = tf.reduce_sum(probs * log_probs, axis=1)
        elif policy_type == 'gaussian':
            assert min_var is not None
            dim_action = tf_logits.shape[1] - 1
            ph_action = tf.placeholder(tf.float32, [None, dim_action])
            self.tf_mean = tf_logits[:, :-1]
            self.tf_var = tf.maximum(tf.nn.softplus(tf_logits[:, -1]), min_var)
            kfac_policy_loss = 'normal_predictive', (self.tf_mean, self.tf_var)
            two_var = 2.0 * self.tf_var
            act_minus_mean = ph_action - self.tf_mean
            log_norm = tf.reduce_sum(act_minus_mean**2, axis=1) / two_var
            log_2pi_var = self.LOGPI + tf.log(two_var)
            log_probs_act = -(log_norm + 0.5 * int(dim_action) * log_2pi_var)
            if entropy_weight:
                neg_entropy = 0.5 * (log_2pi_var + 1.0)
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
        self.ph_train_list = [self.ph_state, ph_action, ph_advantage, ph_target]

    def action_values(self, state):
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def state_value(self, state):
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def ac_values(self, state):
        return self.sess.run([self.tf_logits, self.tf_value],
                             feed_dict={self.ph_state: state})


