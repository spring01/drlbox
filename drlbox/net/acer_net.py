"""Net class for ACER"""
import tensorflow as tf
from drlbox.common.namescope import TF_NAMESCOPE
from drlbox.net.ac_net import ACNet


class ACERNet(ACNet):
    """Class that handles loss/update in ACER. Assumes discrete action for now.
    """

    def set_model(self, model):
        """Set a Keras model"""
        self.model = model
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_logits, self.tf_value = model.outputs

    def set_loss(self, entropy_weight=0.01, kl_weight=0.1, trunc_max=10.0,
                 policy_type=None):
        """Set the ACER loss function
        Args:
            entropy_weight: float; weight of the entropy regularization term
            kl_weight: float; weight of the KL divergence loss term, with
                respect to the Polyak averaging net
            trunc_max: float; maximum allowed value of the truncated
                importance sampling weight
            policy_type: None or 'softmax'; type of the stochastic policy
        """
        with tf.name_scope(TF_NAMESCOPE):
            # sample return and baseline placeholders
            ph_target = tf.placeholder(tf.float32, [None])
            ph_baseline = tf.placeholder(tf.float32, [None])

            if policy_type == 'softmax':
                kfac_policy_loss = 'categorical_predictive', (self.tf_logits,)
                num_act = self.tf_logits.shape[1]
                ph_action = tf.placeholder(tf.int32, [None])
                onehot_act = tf.one_hot(ph_action, depth=num_act)

                # importance sampling weight and trunc
                ph_lratio = tf.placeholder(tf.float32, [None, num_act])
                trunc = tf.minimum(trunc_max, ph_lratio)
                trunc_act = tf.reduce_sum(trunc * onehot_act, axis=1)

                # bootstrapped value placeholder
                ph_boot = tf.placeholder(tf.float32, [None, num_act])

                # log policy
                log_probs = tf.nn.log_softmax(self.tf_logits)

                # policy loss: sample return part
                log_probs_act = tf.reduce_sum(log_probs * onehot_act, axis=1)
                adv_ret = trunc_act * (ph_target - ph_baseline)
                policy_loss_ret = -(adv_ret * log_probs_act)

                # policy loss: bootstrapped value part
                probs = tf.nn.softmax(self.tf_logits)
                probs_c = tf.stop_gradient(probs)
                trunc = tf.maximum(0.0, 1.0 - trunc_max / ph_lratio)
                trunc_prob = trunc * probs_c
                adv_val = trunc_prob * (ph_boot - ph_baseline[:, tf.newaxis])
                policy_loss_val = -tf.reduce_sum(adv_val * log_probs, axis=1)

                # KL (wrt averaged policy net) loss
                if kl_weight:
                    ph_avg_logits = tf.placeholder(tf.float32, [None, num_act])
                    avg_probs = tf.nn.softmax(ph_avg_logits)
                    avg_probs_log_probs = avg_probs * log_probs
                    kl_avg_online = -tf.reduce_sum(avg_probs_log_probs, axis=1)

                # state-action value
                value_act = tf.reduce_sum(self.tf_value * onehot_act, axis=1)

                # entropy
                if entropy_weight:
                    neg_entropy = tf.reduce_sum(probs * log_probs, axis=1)
            else:
                raise ValueError('policy_type must be softmax in ACER for now')

            # value (critic) loss
            value_loss = tf.squared_difference(ph_target, value_act)

            # total loss
            self.tf_loss = policy_loss_ret + policy_loss_val + value_loss
            if kl_weight:
                self.tf_loss += kl_avg_online * kl_weight
            if entropy_weight:
                self.tf_loss += neg_entropy * entropy_weight

            # error for prioritization: critic abs td error
            self.tf_error = tf.abs(ph_target - value_act)

        # kfac loss register
        kfac_value_loss = 'normal_predictive', (self.tf_value,)
        self.kfac_loss_list = [kfac_policy_loss, kfac_value_loss]

        # placeholders
        self.ph_train_list = [self.ph_state, ph_action, ph_lratio,
                              ph_target, ph_boot, ph_baseline]
        if kl_weight:
            self.ph_train_list.append(ph_avg_logits)

    def set_soft_update(self, new_weights, update_ratio):
        """Set the soft update operation for the Polyak averaging net"""
        assign_list = []
        for wt, nwt in zip(self.weights, new_weights):
            upd = (1.0 - update_ratio) * wt + update_ratio * nwt
            assign_list.append(wt.assign(upd))
        self.op_soft_update = tf.group(*assign_list)

    def soft_update(self):
        """Perform the soft update operation for the Polyak averaging net"""
        self.sess.run(self.op_soft_update)

