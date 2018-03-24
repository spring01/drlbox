
import tensorflow as tf
from .ac_net import ACNet


'''
ACER assumes discrete action for now.
'''
class ACERNet(ACNet):

    def set_model(self, model):
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, self.tf_value = model.outputs

    def set_loss(self, entropy_weight=0.01, kl_weight=0.1, trunc_max=10.0,
                 policy_type=None):
        # sample return and baseline placeholders
        ph_sample_return = tf.placeholder(tf.float32, [None])
        ph_baseline = tf.placeholder(tf.float32, [None])

        if policy_type == 'softmax':
            kfac_policy_loss = 'categorical_predictive', (self.tf_logits,)
            num_action = self.tf_logits.shape[1]
            ph_action = tf.placeholder(tf.int32, [None])
            action_onehot = tf.one_hot(ph_action, depth=num_action)

            # importance sampling weight and trunc
            ph_lratio = tf.placeholder(tf.float32, [None, num_action])
            trunc = tf.minimum(trunc_max, ph_lratio)
            trunc_act = tf.reduce_sum(trunc * action_onehot, axis=1)

            # bootstrapped value placeholder
            ph_boot_value = tf.placeholder(tf.float32, [None, num_action])

            # log policy
            log_probs = tf.nn.log_softmax(self.tf_logits)

            # policy loss: sample return part
            log_probs_act = tf.reduce_sum(log_probs * action_onehot, axis=1)
            adv_ret = trunc_act * (ph_sample_return - ph_baseline)
            policy_loss_ret = -(adv_ret * log_probs_act)

            # policy loss: bootstrapped value part
            probs = tf.nn.softmax(self.tf_logits)
            probs_c = tf.stop_gradient(probs)
            trunc_prob = tf.maximum(0.0, 1.0 - trunc_max / ph_lratio) * probs_c
            adv_val = trunc_prob * (ph_boot_value - ph_baseline[:, tf.newaxis])
            policy_loss_val = -tf.reduce_sum(adv_val * log_probs, axis=1)

            # KL (wrt averaged policy net) loss
            ph_avg_logits = tf.placeholder(tf.float32, [None, num_action])
            avg_probs = tf.nn.softmax(ph_avg_logits)
            kl_loss = -kl_weight * tf.reduce_sum(avg_probs * log_probs, axis=1)

            # state-action value
            value_act = tf.reduce_sum(self.tf_value * action_onehot, axis=1)

            # entropy
            if entropy_weight:
                neg_entropy = tf.reduce_sum(probs * log_probs, axis=1)
        else:
            raise ValueError('policy_type must be softmax in ACER, for now')

        # value (critic) loss
        value_loss = tf.squared_difference(ph_sample_return, value_act)

        # total loss
        self.tf_loss = policy_loss_ret + policy_loss_val + value_loss + kl_loss
        if entropy_weight:
            self.tf_loss += neg_entropy * entropy_weight

         # kfac loss register
        kfac_value_loss = 'normal_predictive', (self.tf_value,)
        self.kfac_loss_list = [kfac_policy_loss, kfac_value_loss]

        # placeholders
        self.ph_train_list = [self.ph_state, ph_action, ph_lratio,
            ph_sample_return, ph_boot_value, ph_baseline, ph_avg_logits]

    def set_soft_update(self, new_weights, update_ratio):
        assign_list = []
        for wt, nwt in zip(self.weights, new_weights):
            upd = (1.0 - update_ratio) * wt + update_ratio * nwt
            assign_list.append(wt.assign(upd))
        self.op_soft_update = tf.group(*assign_list)

    def soft_update(self):
        self.sess.run(self.op_soft_update)

