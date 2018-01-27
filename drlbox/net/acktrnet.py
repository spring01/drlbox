
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib.kfac.python.ops.layer_collection import LayerCollection
from .acnet import ACNet
from drlbox.layers.noisy_dense import NoisyDenseIG


NOISY_NOT_REG = 'layer_collection register is not implemented for noisy dense'

class ACKTRNet(ACNet):

    '''
    Called after calling set_loss
    '''
    def build_layer_collection(self):
        lc = LayerCollection()
        for layer in self.model.layers:
            weights = tuple(layer.weights)
            if type(layer) is keras.layers.Dense:
                # There must not be activation if layer is keras.layers.Dense
                lc.register_fully_connected(weights, layer.input, layer.output)
            elif type(layer) is NoisyDenseIG:
                raise NotImplementedError(NOISY_NOT_REG)
            elif type(layer) is keras.layers.Conv2D:
                strides = 1, *layer.strides, 1
                padding = layer.padding.upper()
                lc.register_conv2d(weights, strides, padding,
                                   layer.input, layer.output)
        tf_value, tf_logits = self.model.outputs
        lc.register_normal_predictive_distribution(tf_value)
        if self.action_mode == self.DISCRETE:
            lc.register_categorical_predictive_distribution(tf_logits)
        elif self.action_mode == self.CONTINUOUS:
            mean = self.tf_mean
            var = tf.expand_dims(self.tf_var, -1)
            lc.register_normal_predictive_distribution(mean, var)
        else:
            raise ValueError('model.action_mode not recognized')
        return lc

    def set_optimizer(self, kfac, train_weights=None, inv_update_interval=100):
        self.inv_update_interval = inv_update_interval
        self.train_step_counter = 0
        grads_and_vars = kfac.compute_gradients(self.tf_loss, self.weights)
        if train_weights is None:
            train_weights = self.weights
        grad_op = kfac.apply_gradients(grads_and_vars, train_weights)
        self.op_train = [self.tf_loss, grad_op, kfac.cov_update_op]
        self.op_inv_update = kfac.inv_update_op

    def train_on_batch(self, state, action, advantage, target):
        loss = super().train_on_batch(state, action, advantage, target)
        self.train_step_counter += 1
        if self.train_step_counter >= self.inv_update_interval:
            self.sess.run(self.op_inv_update)
            self.train_step_counter = 0
        return loss

