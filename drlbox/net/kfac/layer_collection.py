"""Rederived LayerCollection in order to handle NoisyNet"""
from tensorflow.contrib.kfac.python.ops.layer_collection import (LayerCollection,
    VARIABLE_SCOPE, APPROX_KRONECKER_NAME, _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES)
from .fisher_blocks import FullyConnectedNoisyNetFGKFACBasicFB


_FULLY_CONNECTED_NOISYNETFG_APPROX_TO_BLOCK_TYPES = {
    APPROX_KRONECKER_NAME: FullyConnectedNoisyNetFGKFACBasicFB,
}


class LayerCollectionExt(LayerCollection):
    """Rederived LayerCollection class that can register a factorized Gaussian
    NoisyNet layer.
    """

    def register_fully_connected_noisynetfg(self,
                                            params_quiet,
                                            params_noise_scale,
                                            inputs,
                                            outputs,
                                            input_noise,
                                            output_noise,
                                            approx=None,
                                            reuse=VARIABLE_SCOPE):
        """Registers a fully connnected factorized Gaussian NoisyNet layer.

        Args:
          params_quiet: Tensor or 2-tuple of Tensors corresponding to non-noisy
            weight and bias of this layer.
            Weight matrix should have shape [input_size, output_size].
            Bias should have shape [output_size].
          params_noise_scale: Tensor or 2-tuple of Tensors corresponding to
            weight noise scales and bias noise scales of this layer.
            Weight noise scales matrix should have shape [input_size, output_size].
            Bias noise scales should have shape [output_size].
          inputs: Tensor of shape [batch_size, input_size]. Inputs to layer.
          outputs: Tensor of shape [batch_size, output_size]. Outputs
            produced by layer.
          input_noise: Tensor of shape [input_size]. Factorized input noises.
          output_noise: Tensor of shape [output_size]. Factorized output noises.
          approx: str. One of "kron" or "diagonal" (Note: "diagonal" is not implemented yet).
          reuse: bool or str.  If True, reuse an existing FisherBlock. If False,
            create a new FisherBlock.  If "VARIABLE_SCOPE", use
            tf.get_variable_scope().reuse.

        Raises:
          ValueError: For improper value to 'approx'.
          KeyError: If reuse == True but no FisherBlock found for 'params_quiet'.
          ValueError: If reuse == True and FisherBlock found but of the wrong type.
        """
        if approx is None:
            approx = self._get_linked_approx(params_quiet)
            if approx is None:
                approx = self.default_fully_connected_approximation # should be 'kron'

        if approx not in _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES:
            raise ValueError("Bad value {} for approx.".format(approx))

        block_type_fc = _FULLY_CONNECTED_APPROX_TO_BLOCK_TYPES[approx]
        block_type_fg = _FULLY_CONNECTED_NOISYNETFG_APPROX_TO_BLOCK_TYPES[approx]
        has_bias = isinstance(params_quiet, (tuple, list))

        block_fc = self.register_block(params_quiet,
                                       block_type_fc(self, has_bias),
                                       reuse=reuse)
        block_fc.register_additional_minibatch(inputs, outputs)

        block_fg = self.register_block(params_noise_scale,
                                       block_type_fg(self, has_bias),
                                       reuse=reuse)
        block_fg.register_additional_minibatch(inputs, outputs,
                                               input_noise, output_noise)


