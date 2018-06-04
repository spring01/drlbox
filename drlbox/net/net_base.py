"""Base class for a trainable neural net for RL"""
import tensorflow as tf
from drlbox.common.namescope import TF_NAMESCOPE
from drlbox.layer.noisy_dense import NoisyDenseIG, NoisyDenseFG


class RLNet:
    """Base class RLNet"""

    op_sync = None          # tf operation of syncing to target weights
    kfac_loss_list = []     # list of losses used in K-FAC

    def set_model(self, model):
        """Set Keras model; to be overloaded in derived classes."""
        raise NotImplementedError

    def set_loss(self, *args, **kwargs):
        """Set loss; to be overloaded in derived classes."""
        raise NotImplementedError

    def action_values(self, state):
        """Return state-action values; to be overloaded by derived classes"""
        raise NotImplementedError

    def state_value(self, *args, **kwargs):
        """Return state-only value; to be overloaded in derived classes."""
        raise NotImplementedError

    def set_session(self, sess):
        """Set TensorFlow session"""
        self.sess = sess

    def set_sync_weights(self, sync_weights):
        """Set sync weight target. 'sync_weight' is a list of tf tensors."""
        with tf.name_scope(TF_NAMESCOPE):
            op_sync_list = [wt.assign(swt)
                            for wt, swt in zip(self.weights, sync_weights)]
            self.op_sync = tf.group(*op_sync_list)

    def set_optimizer(self, opt, clip_norm=None, train_weights=None,
                      priority_type=None, batch_size=None):
        """Set optimizer.
        Args:
            opt: an instance of tf.train.Optimizer
            clip_norm: None means no gradient clipping, otherwise it is
                the maximum norm of gradient clipping, i.e. the second argument
                to be passed to tf.clip_by_global_norm
            train_weights: (list of) neural network weights to be optimized
            priority_type: None, 'error', or 'differential'; priority type in
                prioritized experience replay
            batch_size: batch size for neural network optimization; only useful
                when priority_type == 'differential'
        """
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
        """Set K-FAC optimizer
        Args:
            kfac: an instance of drlbox.net.kfac.optimizer.KfacOptimizerTV
                (derived from tf.contrib.kfac.optimizer.KfacOptimizer)
            inv_upd_interval: interval for updating the inverse of the Fisher
                information matrix
            kwargs: keyword arguments to be passed to
        """
        self.set_optimizer(kfac, **kwargs)
        self.op_train = [self.op_train, kfac.cov_update_op]
        self.op_periodic = kfac.inv_update_op
        self.periodic_interval = inv_upd_interval

    def train_on_batch(self, *args, batch_weight=None):
        """Train the neural network on a batch of rollouts
        Args:
            args: training arguments. Different in each child class.
                e.g., in A3C, args = states, actions, advantages, target_values
                      in DQN, args = states, actions, target_values
            batch_weight: sample weight to be applied on each rollout
        """
        if batch_weight is None:
            # args[0] is input states, len(args[0]) is therefore batch size
            batch_weight = [1.0] * len(args[0])
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

    def sync(self):
        """Sync this net to the preset syncing target"""
        self.sess.run(self.op_sync)

    def set_noise_list(self):
        """Set a list of noise variables if NoisyNet is involved."""
        self.noise_list = []
        for layer in self.model.layers:
            if type(layer) in {NoisyDenseIG, NoisyDenseFG}:
                self.noise_list.extend(layer.noise_list)

    def sample_noise(self):
        """Resample noise variables in NoisyNet."""
        for noise in self.noise_list:
            self.sess.run(noise.initializer)

    def save_model(self, filename):
        """Save the Keras model to filename."""
        self.model.save(filename)

