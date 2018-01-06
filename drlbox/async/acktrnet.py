
import tensorflow as tf
from tensorflow import keras
from tensorflow.contrib import kfac
from .lc_var import LayerCollectionWithVariance
from .acnet import ACNet
from ..feature.preact_layers import DensePreact, Conv2DPreact


class ACKTRNet(ACNet):

    def __init__(self, model, inv_update_interval=100):
        super().__init__(model)
        self.train_step_counter = 0
        self.inv_update_interval = inv_update_interval

    def build_layer_collection(self, model):
        lc = LayerCollectionWithVariance()
        for layer in model.layers:
            weights = tuple(layer.weights)
            if isinstance(layer, DensePreact):
                lc.register_fully_connected(weights, layer.input, layer.preact)
            elif isinstance(layer, keras.layers.Dense):
                # There must not be activation if layer is keras.layers.Dense
                lc.register_fully_connected(weights, layer.input, layer.output)
            elif isinstance(layer, Conv2DPreact):
                strides = 1, *layer.strides, 1
                padding = layer.padding.upper()
                lc.register_conv2d(weights, strides, padding,
                                   layer.input, layer.preact)
        tf_value, tf_logits = model.outputs
        lc.register_normal_predictive_distribution(tf_value)
        if model.action_mode == 'discrete':
            lc.register_categorical_predictive_distribution(tf_logits)
        elif model.action_mode == 'continuous':
            mean, var = tf_logits[:, :-1], tf.nn.softplus(tf_logits[:, -1:])
            lc.register_normal_predictive_distribution_with_variance(mean, var)
        else:
            raise ValueError('model.action_mode not recognized')
        return lc

    def set_optimizer(self, kfac, train_weights=None):
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

