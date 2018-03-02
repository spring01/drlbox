
import tensorflow as tf
from .net_base import RLNet
from drlbox.common.util import discrete_action


class QNet(RLNet):

    def build_model(self, state, feature, action_space):
        if not discrete_action(action_space):
            raise ValueError('action_space must be discrete in DQN')
        flatten = tf.keras.layers.Flatten()
        feature = flatten(feature)
        q_value = self.dense_layer(action_space.n)(feature)
        model = tf.keras.models.Model(inputs=state, outputs=q_value)
        return model

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self):
        ph_action = tf.placeholder(tf.int32, [None])
        action_onehot = tf.one_hot(ph_action, depth=self.tf_values.shape[1])
        ph_target = tf.placeholder(tf.float32, [None])
        act_values = tf.reduce_sum(self.tf_values * action_onehot, axis=1)
        self.tf_loss = tf.losses.huber_loss(ph_target, act_values)
        self.ph_action = ph_action
        self.ph_target = ph_target

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, target):
        feed_dict = {self.ph_state:         state,
                     self.ph_action:        action,
                     self.ph_target:        target,
                     }
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

