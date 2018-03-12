
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base
from tensorflow.python.ops.init_ops import Constant


class NoisyDense(tf.keras.layers.Dense):

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
        scale_init = Constant(value=(0.5 / np.sqrt(kernel_shape[0])))
        kernel_noise_scale = self.add_variable('kernel_noise_scale',
                                               shape=kernel_shape,
                                               initializer=scale_init,
                                               dtype=self.dtype,
                                               trainable=True)
        kernel_noise = self.make_kernel_noise(shape=kernel_shape)
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
                                                 initializer=scale_init,
                                                 dtype=self.dtype,
                                                 trainable=True)
            bias_noise = self.make_bias_noise(shape=bias_shape)
            self.bias = bias_quiet + bias_noise_scale * bias_noise
        else:
            self.bias = None
        self.built = True

    def make_kernel_noise(self, shape):
        raise NotImplementedError

    def make_bias_noise(self, shape):
        raise NotImplementedError


'''
Noisy dense layer with independent Gaussian noise
'''
class NoisyDenseIG(NoisyDense):

    def make_kernel_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        kernel_noise = tf.Variable(noise, trainable=False, dtype=self.dtype)
        self.noise_list = [kernel_noise]
        return kernel_noise

    def make_bias_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        bias_noise = tf.Variable(noise, trainable=False, dtype=self.dtype)
        self.noise_list.append(bias_noise)
        return bias_noise


'''
Noisy dense layer with factorized Gaussian noise
'''
class NoisyDenseFG(NoisyDense):

    def make_kernel_noise(self, shape):
        kernel_noise_input = self.make_fg_noise(shape=[shape[0]])
        kernel_noise_output = self.make_fg_noise(shape=[shape[1]])
        self.noise_list = [kernel_noise_input, kernel_noise_output]
        kernel_noise = kernel_noise_input[:, tf.newaxis] * kernel_noise_output
        return kernel_noise

    def make_bias_noise(self, shape):
        return self.noise_list[1] # kernel_noise_output

    def make_fg_noise(self, shape):
        noise = tf.random_normal(shape, dtype=self.dtype)
        trans_noise = tf.sign(noise) * tf.sqrt(tf.abs(noise))
        return tf.Variable(trans_noise, trainable=False, dtype=self.dtype)


