
from tensorflow.contrib.keras import layers, backend as K

class DensePreact(layers.Dense):

    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        self.preactivation = output
        if self.activation is not None:
            output = self.activation(output)
        return output

