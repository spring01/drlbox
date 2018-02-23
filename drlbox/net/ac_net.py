
import tensorflow as tf
import gym.spaces
from .net_base import RLNet
from drlbox.common.util import discrete_action, continuous_action


class ACNet(RLNet):

    LOGPI               = 1.1447298858494002
    act_decomp_value    = False

    @classmethod
    def from_sfa(cls, state, feature, action_space):
        self = cls()
        flatten = tf.keras.layers.Flatten()
        if type(feature) is tuple:
            assert len(feature) == 2
            # separated logits/value streams when feature is a length 2 tuple
            feature_logits, feature_value = map(flatten, feature)
        else:
            # feature is a single stream otherwise
            feature_logits = feature_value = flatten(feature)
        if discrete_action(action_space):
            self.action_mode = self.DISCRETE
            size_logits = action_space.n
            size_value = size_logits if self.act_decomp_value else 1
            init = tf.keras.initializers.RandomNormal(stddev=1e-3)
        elif continuous_action(action_space):
            self.action_mode = self.CONTINUOUS
            size_logits = len(action_space.shape) + 1
            size_value = 1
            init = 'glorot_uniform'
        else:
            raise ValueError('Invalid type of action_space')
        logits_layer = self.dense_layer(size_logits, kernel_initializer=init)
        logits = logits_layer(feature_logits)
        value = self.dense_layer(size_value)(feature_value)
        model = tf.keras.models.Model(inputs=state, outputs=[logits, value])
        self.set_model(model)
        return self

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, tf_value = model.outputs
        self.tf_value = tf_value[:, 0]

    def set_loss(self, entropy_weight=0.01, min_var=None):
        tf_logits = self.tf_logits
        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])

        if self.action_mode == self.DISCRETE:
            ph_action = tf.placeholder(tf.int32, [None])
            log_probs = tf.nn.log_softmax(tf_logits)
            action_onehot = tf.one_hot(ph_action, depth=tf_logits.shape[1])
            log_probs_act = tf.reduce_sum(log_probs * action_onehot, axis=1)
            if entropy_weight:
                probs = tf.nn.softmax(tf_logits)
                neg_entropy = tf.reduce_sum(probs * log_probs)
        elif self.action_mode == self.CONTINUOUS:
            assert min_var is not None
            dim_action = tf_logits.shape[1] - 1
            ph_action = tf.placeholder(tf.float32, [None, dim_action])
            self.tf_mean = tf_logits[:, :-1]
            self.tf_var = tf.maximum(tf.nn.softplus(tf_logits[:, -1]), min_var)
            two_var = 2.0 * self.tf_var
            act_minus_mean = ph_action - self.tf_mean
            log_norm = tf.reduce_sum(act_minus_mean**2, axis=1) / two_var
            log_2pi_var = self.LOGPI + tf.log(two_var)
            log_probs_act = -(log_norm + 0.5 * int(dim_action) * log_2pi_var)
            if entropy_weight:
                neg_entropy = 0.5 * tf.reduce_sum(log_2pi_var + 1.0)
        else:
            raise ValueError('Invalid action_mode')

        policy_loss = -tf.reduce_sum(log_probs_act * ph_advantage)
        value_squared_diff = tf.squared_difference(ph_target, self.tf_value)
        value_loss = tf.reduce_sum(value_squared_diff)
        self.tf_loss = policy_loss + value_loss
        if entropy_weight:
            self.tf_loss += neg_entropy * entropy_weight

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
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

