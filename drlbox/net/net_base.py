
import tensorflow as tf
from drlbox.common.namescope import TF_NAMESCOPE
from drlbox.layer.noisy_dense import NoisyDenseIG, NoisyDenseFG


class RLNet:

    op_sync = None
    kfac_loss_list = []

    # net constructed by set_model only can predict but cannot be trained
    def set_model(self, model):
        raise NotImplementedError

    def set_session(self, sess):
        self.sess = sess

    def set_sync_weights(self, sync_weights):
        with tf.name_scope(TF_NAMESCOPE):
            op_sync_list = [wt.assign(swt)
                            for wt, swt in zip(self.weights, sync_weights)]
            self.op_sync = tf.group(*op_sync_list)

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer(self, opt, clip_norm=None, train_weights=None,
                      priority_type=None, batch_size=None):
        with tf.name_scope(TF_NAMESCOPE):
            self.ph_batch_weight = tf.placeholder(tf.float32, [None])
            tf_batch_loss = tf.reduce_sum(self.tf_loss * self.ph_batch_weight)
            grads_and_vars = opt.compute_gradients(tf_batch_loss, self.weights)
            grads = [g for g, v in grads_and_vars]
            if clip_norm is not None:
                grads, _ = tf.clip_by_global_norm(grads, clip_norm)
            if train_weights is None:
                train_weights = self.weights
            self.op_train = opt.apply_gradients(zip(grads, train_weights))
            self.op_result = [tf_batch_loss]
            self.op_periodic = []
            self.periodic_interval = None
            self.periodic_counter = 0

            # op_result should include priority term if requested
            if priority_type is not None:
                if priority_type == 'error':
                    tf_batch_priority = [tf.reduce_mean(error)
                        for error in tf.split(self.tf_error, batch_size)]
                elif priority_type == 'differential':
                    tf_batch_priority = []
                    for loss in tf.split(self.tf_loss, batch_size):
                        grad_list = tf.gradients(loss, self.model.outputs)
                        norm = sum(tf.norm(grad, ord=1) for grad in grad_list)
                        tf_batch_priority.append(norm)
                else:
                    message = 'priority_type={} invalid'.format(priority_type)
                    raise ValueError(message)
                self.op_result.append(tf.stack(tf_batch_priority))

    def set_kfac(self, kfac, inv_upd_interval, **kwargs):
        self.set_optimizer(kfac, **kwargs)
        self.op_train = [self.op_train, kfac.cov_update_op]
        self.op_periodic = kfac.inv_update_op
        self.periodic_interval = inv_upd_interval

    def action_values(self, state):
        raise NotImplementedError

    def train_on_batch(self, *args, batch_weight=None):
        if batch_weight is None:
            batch_weight = [1.0] * len(args[0])     # trick to get batch size
        feed_dict = {ph: arg for ph, arg in zip(self.ph_train_list, args)}
        feed_dict[self.ph_batch_weight] = batch_weight

        # run training and result in order
        self.sess.run(self.op_train, feed_dict=feed_dict)
        result = self.sess.run(self.op_result, feed_dict=feed_dict)
        if self.periodic_interval is not None:
            if self.periodic_counter >= self.periodic_interval:
                self.sess.run(self.op_periodic)
                self.periodic_counter = 0
            self.periodic_counter += 1
        return result

    def state_value(self, *args, **kwargs):
        raise NotImplementedError

    def sync(self):
        self.sess.run(self.op_sync)

    def set_noise_list(self):
        self.noise_list = []
        for layer in self.model.layers:
            if type(layer) in {NoisyDenseIG, NoisyDenseFG}:
                self.noise_list.extend(layer.noise_list)

    def sample_noise(self):
        for noise in self.noise_list:
            self.sess.run(noise.initializer)

    def save_model(self, filename):
        self.model.save(filename)

