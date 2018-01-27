
import h5py
import tensorflow as tf


class RLNet:

    op_sync = None
    DISCRETE, CONTINUOUS = 'discrete', 'continuous' # action mode names
    dense_layer = tf.keras.layers.Dense

    def set_session(self, sess):
        self.sess = sess

    def set_sync_weights(self, sync_weights):
        zip_weights = zip(self.weights, sync_weights)
        self.op_sync = tf.group(*[wt.assign(swt) for wt, swt in zip_weights])

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer(self, optimizer, clip_norm=None, train_weights=None):
        grads_and_vars = optimizer.compute_gradients(self.tf_loss, self.weights)
        grads = [g for g, v in grads_and_vars]
        if clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        if train_weights is None:
            train_weights = self.weights
        op_grad = optimizer.apply_gradients(zip(grads, train_weights))
        self.op_train = [self.tf_loss, op_grad]

    def action_values(self, state):
        raise NotImplementedError

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError

    def state_value(self, *args, **kwargs):
        raise NotImplementedError

    def sync(self):
        self.sess.run(self.op_sync)

    def save_weights(self, filename):
        with h5py.File(filename, 'w') as save:
            for idx, value in enumerate(self.sess.run(self.weights)):
                save.create_dataset(name=str(idx), data=value)

    def load_weights(self, filename):
        saved_weights = []
        with h5py.File(filename, 'r') as save:
            for idx in range(len(self.weights)):
                saved_weights.append(save[str(idx)][...])
        zip_weights = zip(self.weights, saved_weights)
        self.sess.run(tf.group(*[wt.assign(swt) for wt, swt in zip_weights]))

