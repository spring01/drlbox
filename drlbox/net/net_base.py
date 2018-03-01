
import tensorflow as tf
from drlbox.layer.noisy_dense import NoisyDenseIG


class RLNet:

    op_sync = None
    DISCRETE, CONTINUOUS = 'discrete', 'continuous' # action mode names
    dense_layer = tf.keras.layers.Dense

    def build_model(self, state, feature, action_space):
        raise NotImplementedError

    # net constructed by set_model only can predict but cannot be trained
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

    def set_noise_list(self):
        self.noise_list = []
        for layer in self.model.layers:
            if type(layer) is NoisyDenseIG:
                self.noise_list.extend(layer.noise_list())

    def sample_noise(self):
        for noise in self.noise_list:
            self.sess.run(noise.initializer)

    def save_model(self, filename):
        self.model.save(filename)

