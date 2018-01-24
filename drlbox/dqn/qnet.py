
import tensorflow as tf
from ..common.rlnet import RLNet


class QNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self):
        tf_values = self.tf_values
        batch_size, num_actions = tf_values.shape
        ph_target = tf.placeholder(tf.float32, [batch_size, num_actions])
        ph_weight = tf.placeholder(tf.float32, [batch_size])
        weight_tile = tf.tile(tf.expand_dims(ph_weight, 1), [1, num_actions])
        self.tf_loss = tf.losses.huber_loss(ph_target, tf_values, weight_tile,
            reduction=tf.losses.Reduction.MEAN)
        self.ph_target = ph_target
        self.ph_sample_weight = ph_weight

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, target, sample_weight=None):
        if sample_weight is None:
            sample_weight = [1.0] * len(state)
        feed_dict = {self.ph_state:         state,
                     self.ph_target:        target,
                     self.ph_sample_weight: sample_weight}
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

