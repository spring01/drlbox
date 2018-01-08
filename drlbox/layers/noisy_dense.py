
import tensorflow as tf
from tensorflow.python.keras.layers import Dense
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops.init_ops import Constant
from .preact_layers import embed_preact


IG_SCALE_INIT = 0.017

'''
Noisy dense layer with independent Gaussian noise
'''
class NoisyDenseIG(Dense):

    def build(self, input_shape):
        input_shape = tensor_shape.TensorShape(input_shape)
        if input_shape[-1].value is None:
            raise ValueError('The last dimension of the inputs to `Dense` '
                             'should be defined. Found `None`.')
        self.input_spec = base.InputSpec(min_ndim=2,
                                         axes={-1: input_shape[-1].value})
        kernel_shape = [input_shape[-1].value, self.units]
        kernel_quiet = self.add_variable('kernel_quiet',
                                         shape=kernel_shape,
                                         initializer=self.kernel_initializer,
                                         regularizer=self.kernel_regularizer,
                                         constraint=self.kernel_constraint,
                                         dtype=self.dtype,
                                         trainable=True)
        kernel_noise_scale = self.add_variable('kernel_noise_scale',
                                               shape=kernel_shape,
                                               initializer=Constant(value=IG_SCALE_INIT),
                                               dtype=self.dtype,
                                               trainable=True)
        kernel_noise = tf.random_normal(kernel_shape, mean=0.0, stddev=1.0, dtype=self.dtype)
        self.kernel = kernel_quiet + kernel_noise_scale * kernel_noise
        if self.use_bias:
            bias_shape = [self.units,]
            bias_quiet = self.add_variable('bias_quiet',
                                           shape=bias_shape,
                                           initializer=self.bias_initializer,
                                           regularizer=self.bias_regularizer,
                                           constraint=self.bias_constraint,
                                           dtype=self.dtype,
                                           trainable=True)
            bias_noise_scale = self.add_variable(name='bias_noise_scale',
                                                 shape=bias_shape,
                                                 initializer=Constant(value=IG_SCALE_INIT),
                                                 dtype=self.dtype,
                                                 trainable=True)
            bias_noise = tf.random_normal(bias_shape, mean=0.0, stddev=1.0, dtype=self.dtype)
            self.bias = bias_quiet + bias_noise_scale * bias_noise
        else:
            self.bias = None
        self.built = True

    def call(self, inputs):
        return embed_preact(self, super(), inputs)


