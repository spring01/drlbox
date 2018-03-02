
from tensorflow import keras
from tensorflow.contrib import kfac
from drlbox.layer.noisy_dense import NoisyDenseIG


class KfacOptimizerTV(kfac.optimizer.KfacOptimizer):

    def apply_gradients(self, grads_and_vars, *args, **kwargs):
        """Applies gradients to variables.
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
          *args: Additional arguments for super.apply_gradients.
          **kwargs: Additional keyword arguments for super.apply_gradients.
        Returns:
          An `Operation` that applies the specified gradients.
        """
        grads, train_vars = zip(*grads_and_vars)
        grads_and_vars = list(zip(grads, self.variables))

        # Compute step.
        steps_and_vars = self._compute_update_steps(grads_and_vars)

        # Modify variables in train_vars instead of grads_and_vars
        zip_st = zip(steps_and_vars, train_vars)
        steps_and_vars = [(step, t_var) for (step, _), t_var in zip_st]

        # Update trainable variables with this step.
        # Note: this super is getting the grandparent class of this class.
        super_obj = super(kfac.optimizer.KfacOptimizer, self)
        return super_obj.apply_gradients(steps_and_vars, *args, **kwargs)


NOISY_NOT_REG = 'layer_collection register is not implemented for NoisyDenseIG'

def build_layer_collection(layer_list, loss_list):
    lc = kfac.layer_collection.LayerCollection()

    # register layers
    for layer in layer_list:
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

    # register losses
    for loss_type, args in loss_list:
        if loss_type == 'normal_predictive':
            reg_func = lc.register_normal_predictive_distribution
        elif loss_type == 'categorical_predictive':
            reg_func = lc.register_categorical_predictive_distribution
        reg_func(*args)

    return lc


