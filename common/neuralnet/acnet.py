
import tensorflow as tf
from .rlnet import RLNet


class ACNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        tf_value, self.tf_logits = model.outputs
        self.tf_value = tf_value[:, 0]

    def set_loss(self, entropy_weight=0.01):
        tf_logits = self.tf_logits
        log_probs = tf.nn.log_softmax(tf_logits)
        probs = tf.nn.softmax(tf_logits)

        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])
        ph_action = tf.placeholder(tf.float32, tf_logits.shape.as_list())

        log_probs_act = tf.reduce_sum(log_probs * ph_action, axis=1)
        policy_loss = -tf.reduce_sum(log_probs_act * ph_advantage)
        value_loss = tf.nn.l2_loss(self.tf_value - ph_target)
        entropy = -tf.reduce_sum(probs * log_probs)
        self.tf_loss = policy_loss + value_loss - entropy * entropy_weight
        self.ph_advantage = ph_advantage
        self.ph_target = ph_target
        self.ph_action = ph_action

    def action_values(self, state):
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def state_value(self, state):
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, advantage, target):
        feed_dict = {self.ph_state:     state,
                     self.ph_action:    action,
                     self.ph_advantage: advantage,
                     self.ph_target:    target}
        self.sess.run(self.op_train, feed_dict=feed_dict)

