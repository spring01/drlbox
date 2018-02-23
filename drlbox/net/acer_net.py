
import tensorflow as tf
import gym.spaces
from .ac_net import ACNet


'''
ACER assumes discrete action for now.
'''
class ACERNet(ACNet):

    act_decomp_value = True

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, self.tf_value = model.outputs

    def set_loss(self, entropy_weight=0.01, kl_weight=0.1, trunc_max=10.0):
        num_action = self.tf_logits.shape[1]

        if self.action_mode == self.DISCRETE:
            ph_action = tf.placeholder(tf.int32, [None])
            action_onehot = tf.one_hot(ph_action, depth=num_action)

            # importance sampling weight and trunc
            ph_lratio = tf.placeholder(tf.float32, [None, num_action])
            trunc = tf.minimum(trunc_max, ph_lratio)
            trunc_act = tf.reduce_sum(trunc * action_onehot, axis=1)

            # return and value placeholders
            ph_q_ret = tf.placeholder(tf.float32, [None])
            ph_q_val = tf.placeholder(tf.float32, [None, num_action])
            ph_baseline = tf.placeholder(tf.float32, [None])

            # log policy
            log_probs = tf.nn.log_softmax(self.tf_logits)

            # policy loss: sampled return
            log_probs_act = tf.reduce_sum(log_probs * action_onehot, axis=1)
            adv_ret = trunc_act * (ph_q_ret - ph_baseline)
            policy_loss_ret = -tf.reduce_sum(adv_ret * log_probs_act)

            # policy loss: bootstrapped value
            probs = tf.nn.softmax(self.tf_logits)
            probs_c = tf.stop_gradient(probs)
            trunc_prob = tf.maximum(0.0, 1.0 - trunc_max / ph_lratio) * probs_c
            adv_val = trunc_prob * (ph_q_val - ph_baseline[:, tf.newaxis])
            policy_loss_boot = -tf.reduce_sum(adv_val * log_probs)

            # KL (wrt averaged policy net) loss
            ph_avg_logits = tf.placeholder(tf.float32, [None, num_action])
            avg_probs = tf.nn.softmax(ph_avg_logits)
            kl_loss = -kl_weight * tf.reduce_sum(avg_probs * log_probs)
        else:
            raise ValueError('action_space must be discrete in ACER')

        # value (critic) loss
        value_act = tf.reduce_sum(self.tf_value * action_onehot, axis=1)
        value_squared_diff = tf.squared_difference(ph_q_ret, value_act)
        value_loss = tf.reduce_sum(value_squared_diff)

        # total loss
        self.tf_loss = policy_loss_ret + policy_loss_boot + value_loss + kl_loss

        # entropy
        if entropy_weight:
            self.tf_loss += tf.reduce_sum(probs * log_probs) * entropy_weight

        # placeholders
        self.ph_action = ph_action
        self.ph_lratio = ph_lratio
        self.ph_q_ret = ph_q_ret
        self.ph_q_val = ph_q_val
        self.ph_baseline = ph_baseline
        self.ph_avg_logits = ph_avg_logits

    def ac_values(self, state):
        return self.sess.run([self.tf_logits, self.tf_value],
                             feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, lratio, q_ret, q_val,
                       baseline, avg_logits):
        feed_dict = {self.ph_state:         state,
                     self.ph_action:        action,
                     self.ph_lratio:        lratio,
                     self.ph_q_ret:         q_ret,
                     self.ph_q_val:         q_val,
                     self.ph_baseline:      baseline,
                     self.ph_avg_logits:    avg_logits}
        loss = self.sess.run(self.op_train, feed_dict=feed_dict)[0]
        return loss

    def set_soft_update(self, new_weights, update_ratio):
        assign_list = []
        for wt, nwt in zip(self.weights, new_weights):
            upd = (1.0 - update_ratio) * wt + update_ratio * nwt
            assign_list.append(wt.assign(upd))
        self.op_soft_update = tf.group(*assign_list)

    def soft_update(self):
        self.sess.run(self.op_soft_update)

