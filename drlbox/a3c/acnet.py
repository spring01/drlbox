
import tensorflow as tf
from ..common.rlnet import RLNet


class ACNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        tf_value, self.tf_logits = model.outputs
        self.tf_value = tf_value[:, 0]

    def set_loss(self, entropy_weight=0.01):
        tf_logits = self.tf_logits
        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])

        # discrete action
        log_probs = tf.nn.log_softmax(tf_logits)
        probs = tf.nn.softmax(tf_logits)
        ph_action = tf.placeholder(tf.int32, [None])
        action_1h = tf.one_hot(ph_action, depth=tf_logits.shape[1])
        log_probs_act = tf.reduce_sum(log_probs * action_1h, axis=1)

        neg_policy_loss = tf.reduce_sum(log_probs_act * ph_advantage)
        value_loss = tf.nn.l2_loss(self.tf_value - ph_target)
        neg_entropy = tf.reduce_sum(probs * log_probs) * entropy_weight
        self.tf_loss = value_loss - neg_policy_loss + neg_entropy
        self.ph_advantage = ph_advantage
        self.ph_target = ph_target
        self.ph_action = ph_action

    def action_values(self, state):
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def state_value(self, state):
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def state_loss(self, state, action, advantage, target):
        feed_dict = {self.ph_state:     state,
                     self.ph_action:    action,
                     self.ph_advantage: advantage,
                     self.ph_target:    target}
        return self.sess.run(self.tf_loss, feed_dict=feed_dict)

    def train_on_batch(self, state, action, advantage, target):
        feed_dict = {self.ph_state:     state,
                     self.ph_action:    action,
                     self.ph_advantage: advantage,
                     self.ph_target:    target}
        self.sess.run(self.op_train, feed_dict=feed_dict)

