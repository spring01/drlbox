
import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.contrib.kfac.python.ops import estimator as est
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
KfacOptimizer = tf.contrib.kfac.optimizer.KfacOptimizer


class KfacOptimizerTV(KfacOptimizer):

    def apply_gradients(self, grads_and_vars, train_vars, *args, **kwargs):
        """Applies gradients to variables.
        Args:
          grads_and_vars: List of (gradient, variable) pairs.
          *args: Additional arguments for super.apply_gradients.
          **kwargs: Additional keyword arguments for super.apply_gradients.
        Returns:
          An `Operation` that applies the specified gradients.
        """
        # In Python 3, grads_and_vars can be a zip() object which can only be
        # iterated over once. By converting it to a list, we ensure that it can be
        # iterated over more than once.
        grads_and_vars = list(grads_and_vars)

        # Compute step.
        steps_and_vars = self._compute_update_steps(grads_and_vars)

        # Modify variables in train_vars instead of grads_and_vars
        zip_st = zip(steps_and_vars, train_vars)
        steps_and_vars = [(step, t_var) for (step, _), t_var in zip_st]

        # Update trainable variables with this step.
        return super(KfacOptimizer, self).apply_gradients(steps_and_vars, *args,
                                                          **kwargs)

