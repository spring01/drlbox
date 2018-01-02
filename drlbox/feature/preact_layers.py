
import tensorflow as tf


class DensePreact(tf.keras.layers.Dense):

    def call(self, inputs):
        return _embed_preact(self, super(), inputs)


class Conv2DPreact(tf.keras.layers.Conv2D):

    def call(self, inputs):
        return _embed_preact(self, super(), inputs)


def _embed_preact(self, super_obj, inputs):
    activation = self.activation
    self.activation = None
    output = super_obj.call(inputs)
    self.preact = output
    if activation is not None:
        output = activation(output)
    self.activation = activation
    return output

