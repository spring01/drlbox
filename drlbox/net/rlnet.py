
import h5py
import tensorflow as tf


class RLNet:

    op_sync = None
    DISCRETE, CONTINUOUS = 'discrete', 'continuous' # action mode names
    dense_layer = tf.keras.layers.Dense

    # net constructed by from_model can predict but cannot be trained
    @classmethod
    def from_model(cls, model):
        self = cls()
        self.set_model(model)
        return self

    def set_model(self, model):
        raise NotImplementedError

    def set_session(self, sess):
        self.sess = sess

    def set_sync_weights(self, sync_weights):
        zip_weights = zip(self.weights, sync_weights)
        self.op_sync = tf.group(*[wt.assign(swt) for wt, swt in zip_weights])

    def set_loss(self, *args, **kwargs):
        raise NotImplementedError

    def set_optimizer(self, learning_rate, epsilon, clip_norm=None,
                      train_weights=None):
        adam = tf.train.AdamOptimizer(learning_rate, epsilon=epsilon)
        grads_and_vars = adam.compute_gradients(self.tf_loss, self.weights)
        grads = [g for g, v in grads_and_vars]
        if clip_norm is not None:
            grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        if train_weights is None:
            train_weights = self.weights
        op_grad = adam.apply_gradients(zip(grads, train_weights))
        self.op_train = [self.tf_loss, op_grad]

    def action_values(self, state):
        raise NotImplementedError

    def train_on_batch(self, *args, **kwargs):
        raise NotImplementedError

    def state_value(self, *args, **kwargs):
        raise NotImplementedError

    def sync(self):
        self.sess.run(self.op_sync)

    def save_model(self, filename):
        self.model.save(filename)

    @staticmethod
    def load_model(filename, custom_objects=None):
        return tf.keras.models.load_model(filename, custom_objects)

