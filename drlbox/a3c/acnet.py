
import numpy as np
import tensorflow as tf
from ..common.rlnet import RLNet


LOG2PI = np.log(2.0 * np.pi)

class ACNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        tf_value, self.tf_logits = model.outputs
        self.tf_value = tf_value[:, 0]
        self.action_mode = model.action_mode

    def set_loss(self, entropy_weight=0.01):
        tf_logits = self.tf_logits
        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])

        if self.action_mode == 'discrete':
            ph_act = tf.placeholder(tf.int32, [None])
            log_probs = tf.nn.log_softmax(tf_logits)
            probs = tf.nn.softmax(tf_logits)
            action_1h = tf.one_hot(ph_act, depth=tf_logits.shape[1])
            log_probs_act = tf.reduce_sum(log_probs * action_1h, axis=1)
            neg_entropy = tf.reduce_sum(probs * log_probs) * entropy_weight
        elif self.action_mode == 'continuous':
            dim_action = tf_logits.shape[1] - 1
            ph_act = tf.placeholder(tf.float32, [None, dim_action])
            mean = tf_logits[:, :-1]
            var = tf.nn.softplus(tf_logits[:, -1:])
            log_norm = tf.reduce_sum((ph_act - mean)**2 / (2.0 * var), axis=1)
            log_2pi_log_var = LOG2PI + tf.log(var)
            factor = tf.reduce_sum(log_2pi_log_var, axis=1)
            factor *= 0.5 * int(dim_action)
            log_probs_act = -(log_norm + factor)
            neg_entropy = tf.reduce_sum(log_2pi_log_var + 1.0)
            neg_entropy *= entropy_weight * 0.5
        else:
            raise ValueError('action_mode not recognized')

        neg_policy_loss = tf.reduce_sum(log_probs_act * ph_advantage)
        value_loss = tf.nn.l2_loss(self.tf_value - ph_target)
        self.tf_loss = value_loss - neg_policy_loss + neg_entropy
        self.ph_advantage = ph_advantage
        self.ph_target = ph_target
        self.ph_action = ph_act

    def action_values(self, state):
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def state_value(self, state):
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, advantage, target):
        feed_dict = {self.ph_state:     state,
                     self.ph_action:    action,
                     self.ph_advantage: advantage,
                     self.ph_target:    target}
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

