
import tensorflow as tf
from drlbox.common.namescope import TF_NAMESCOPE
from drlbox.net.net_base import RLNet


class QNet(RLNet):

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self):
        with tf.name_scope(TF_NAMESCOPE):
            ph_action = tf.placeholder(tf.int32, [None])
            onehot_act = tf.one_hot(ph_action, depth=self.tf_values.shape[1])
            ph_target = tf.placeholder(tf.float32, [None])
            value_act = tf.reduce_sum(self.tf_values * onehot_act, axis=1)

            # loss
            self.tf_loss = tf.losses.huber_loss(ph_target, value_act,
                reduction=tf.losses.Reduction.NONE)

            # error for prioritization: abs td error
            self.tf_error = tf.abs(ph_target - value_act)

        # kfac loss list
        self.kfac_loss_list = [('normal_predictive', (self.tf_values,))]

        # placeholder list
        self.ph_train_list = [self.ph_state, ph_action, ph_target]

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

