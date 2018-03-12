
from tensorflow.contrib.kfac.python.ops import fisher_blocks as fb
from tensorflow.contrib.kfac import fisher_factors

class FullyConnectedNoisyNetFGKFACBasicFB(fb.FullyConnectedKFACBasicFB):
    """K-FAC FisherBlock for fully-connected (dense) factorized Gaussian noisynet layers.
    This uses the Kronecker-factorized approximation from the original
    K-FAC paper (https://arxiv.org/abs/1503.05671)
    """

    def __init__(self, layer_collection, has_bias=False):
        """Creates a FullyConnectedNoisyNetFGKFACBasicFB block.
        Args:
          layer_collection: The collection of all layers in the K-FAC approximate
              Fisher information matrix to which this FisherBlock belongs.
          has_bias: Whether the component Kronecker factors have an additive bias.
              (Default: False)
        """
        self._inputs = []
        self._outputs = []
        self._input_noise = []
        self._output_noise = []
        self._has_bias = has_bias

        super(fb.FullyConnectedKFACBasicFB, self).__init__(layer_collection)

    def instantiate_factors(self, grads_list, damping):
        """Instantiate Kronecker Factors for this FisherBlock.
        Args:
          grads_list: List of list of Tensors. grads_list[i][j] is the
            gradient of the loss with respect to 'outputs' from source 'i' and
            tower 'j'. Each Tensor has shape [tower_minibatch_size, output_size].
          damping: 0-D Tensor or float. 'damping' * identity is approximately added
            to this FisherBlock's Fisher approximation.
        """
        # TODO(b/68033310): Validate which of,
        #   (1) summing on a single device (as below), or
        #   (2) on each device in isolation and aggregating
        # is faster.
        input_noise = fb._concat_along_batch_dim(self._input_noise)
        output_noise = fb._concat_along_batch_dim(self._output_noise)
        inputs = fb._concat_along_batch_dim(self._inputs) * input_noise

        grads_list = tuple(fb._concat_along_batch_dim(grads) * output_noise
                           for grads in grads_list)

        self._input_factor = self._layer_collection.make_or_get_factor(  #
            fisher_factors.FullyConnectedKroneckerFactor,  #
            ((inputs,), self._has_bias))
        self._output_factor = self._layer_collection.make_or_get_factor(  #
            fisher_factors.FullyConnectedKroneckerFactor,  #
            (grads_list,))
        self._register_damped_input_and_output_inverses(damping)

    def register_additional_minibatch(self, inputs, outputs,
                                      input_noise, output_noise):
        """Registers an additional minibatch to the FisherBlock.
        Args:
          inputs: Tensor of shape [batch_size, input_size]. Inputs to the
            matrix-multiply.
          outputs: Tensor of shape [batch_size, output_size]. Layer preactivations.
          input_noise: Tensor of shape [input_size]. Factorized input noises.
          output_noise: Tensor of shape [output_size]. Factorized output noises.
        """
        self._inputs.append(inputs)
        self._outputs.append(outputs)
        self._input_noise.append(input_noise)
        self._output_noise.append(output_noise)


