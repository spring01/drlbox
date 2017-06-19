
import pickle
import tensorflow as tf


class RLNet(object):

    def set_session(self, sess):
        self.sess = sess

    def set_sync_weights(self, sync_weights):
        zip_weights = zip(self.weights, sync_weights)
        self.op_sync = tf.group(*[wt.assign(swt) for wt, swt in zip_weights])

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer(self, optimizer, clip_norm=40.0, train_weights=None):
        weights = self.weights
        grads = tf.gradients(self.tf_loss, weights)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        if train_weights is None:
            train_weights = weights
        grads_and_vars = list(zip(grads, train_weights))
        self.op_train = optimizer.apply_gradients(grads_and_vars)

    def action_values(self, state):
        raise NotImplementedError

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError

    def sync(self):
        return self.sess.run(self.op_sync)

    def save_weights(self, filename):
        with open(filename, 'wb') as pic:
            pickle.dump(self.sess.run(self.weights), pic)

