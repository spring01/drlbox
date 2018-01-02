
import tensorflow as tf


class DensePreact(tf.keras.layers.Dense):

    def call(self, inputs):
        output = tf.matmul(inputs, self.kernel)
        if self.use_bias:
            output = tf.nn.bias_add(output, self.bias)
        self.preactivation = output
        if self.activation is not None:
            output = self.activation(output)
        return output

