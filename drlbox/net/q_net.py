
import tensorflow as tf
from .net_base import RLNet
from drlbox.common.util import discrete_action


class QNet(RLNet):

    @classmethod
    def from_sfa(cls, state, feature, action_space):
        self = cls()
        if not discrete_action(action_space):
            raise ValueError('action_space must be discrete in Q network')
        q_value = self.dense_layer(action_space.n)(feature)
        model = tf.keras.models.Model(inputs=state, outputs=q_value)
        self.set_model(model)
        return self

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self):
        ph_action = tf.placeholder(tf.int32, [None])
        action_onehot = tf.one_hot(ph_action, depth=self.tf_values.shape[1])
        ph_target = tf.placeholder(tf.float32, [None])
        ph_weight = tf.placeholder(tf.float32, [None])
        act_values = tf.reduce_sum(self.tf_values * action_onehot, axis=1)
        self.tf_loss = tf.losses.huber_loss(ph_target, act_values, ph_weight,
            reduction=tf.losses.Reduction.MEAN)
        self.ph_action = ph_action
        self.ph_target = ph_target
        self.ph_sample_weight = ph_weight

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, target, sample_weight=None):
        if sample_weight is None:
            sample_weight = [1.0] * len(state)
        feed_dict = {self.ph_state:         state,
                     self.ph_action:        action,
                     self.ph_target:        target,
                     self.ph_sample_weight: sample_weight}
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

