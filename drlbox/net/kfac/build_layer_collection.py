"""Build a layer collection object as needed by the K-FAC optimizer."""
from tensorflow import keras
from drlbox.layer.noisy_dense import NoisyDenseIG, NoisyDenseFG
from .layer_collection import LayerCollectionExt


NOISY_NOT_REG = 'layer_collection register is not implemented for NoisyDenseIG'

def build_layer_collection(layer_list, loss_list):
    """Returns a layer collection object
    Args:
        layer_list: a list of Keras layers (usually model.layers)
        loss_list: a list of tuples, each tuple is of form (loss_type, args),
            where loss_type is a string with choices 'normal_predictive' or
            'categorical_predictive', and args is the input arguments to
            the "loss-registration" functions in the LayerCollection class.
    """
    lc = LayerCollectionExt()

    # register layers
    for layer in layer_list:
        weights = tuple(layer.weights)
        if type(layer) is keras.layers.Dense:
            # There must not be activation if layer is keras.layers.Dense
            lc.register_fully_connected(weights, layer.input, layer.output)
        elif type(layer) is NoisyDenseIG:
            raise NotImplementedError(NOISY_NOT_REG)
        elif type(layer) is NoisyDenseFG:
            if layer.use_bias:
                params_quiet = weights[0], weights[2]
                params_noise_scale = weights[1], weights[3]
            else:
                params_quiet = weights[0]
                params_noise_scale = weights[1]
            lc.register_fully_connected_noisynetfg(
                params_quiet=params_quiet,
                params_noise_scale=params_noise_scale,
                inputs=layer.input,
                outputs=layer.output,
                input_noise=layer.noise_list[0],
                output_noise=layer.noise_list[1],
                )
        elif type(layer) is keras.layers.Conv2D:
            strides = 1, *layer.strides, 1
            padding = layer.padding.upper()
            lc.register_conv2d(weights, strides, padding,
                               layer.input, layer.output)

    # register losses
    for loss_type, args in loss_list:
        if loss_type == 'normal_predictive':
            lc.register_normal_predictive_distribution(*args)
        elif loss_type == 'categorical_predictive':
            lc.register_categorical_predictive_distribution(*args)
        else:
            raise ValueError('loss type {} not supported'.format(loss_type))

    return lc
