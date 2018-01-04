
import tensorflow as tf
from tensorflow.python.ops import variables as tf_variables
from tensorflow.contrib.kfac.python.ops import estimator as est
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
KfacOptimizer = tf.contrib.kfac.optimizer.KfacOptimizer


class KfacOptimizerTV(KfacOptimizer):

    def __init__(
          self,
          learning_rate,
          cov_ema_decay,
          damping,
          layer_collection,
          var_list=None,
          momentum=0.,
          momentum_type="regular",
          norm_constraint=None,
          name="KFAC",):
        """Initializes the KFAC optimizer with the given settings.
        Args:
          learning_rate: The base learning rate for the optimizer.  Should probably
              be set to 1.0 when using momentum_type = 'qmodel', but can still be
              set lowered if desired (effectively lowering the trust in the
              quadratic model.)
          cov_ema_decay: The decay factor used when calculating the covariance
              estimate moving averages.
          damping: The damping factor used to stabilize training due to errors in
              the local approximation with the Fisher information matrix, and to
              regularize the update direction by making it closer to the gradient.
              (Higher damping means the update looks more like a standard gradient
              update - see Tikhonov regularization.)
          layer_collection: The layer collection object, which holds the fisher
              blocks, kronecker factors, and losses associated with the
              graph.  The layer_collection cannot be modified after KfacOptimizer's
              initialization.
          momentum: The momentum value for this optimizer. Only applies when
              momentum_type is 'regular' or 'adam'. (Default: 0)
          momentum_type: The type of momentum to use in this optimizer, one of
              'regular', 'adam', or 'qmodel'. (Default: 'regular')
          norm_constraint: float or Tensor. If specified, the update is scaled down
              so that its approximate squared Fisher norm v^T F v is at most the
              specified value. May only be used with momentum type 'regular'.
              (Default: None)
          name: The name for this optimizer. (Default: 'KFAC')
        Raises:
          ValueError: If the momentum type is unsupported.
          ValueError: If clipping is used with momentum type other than 'regular'.
          ValueError: If no losses have been registered with layer_collection.
          ValueError: If momentum is non-zero and momentum_type is not 'regular'
              or 'adam'.
        """

        variables = var_list
        if variables is None:
          variables = tf_variables.trainable_variables()

        self._fisher_est = est.FisherEstimator(variables, cov_ema_decay, damping,
                                               layer_collection)

        momentum_type = momentum_type.lower()
        legal_momentum_types = ["regular", "adam", "qmodel"]

        if momentum_type not in legal_momentum_types:
          raise ValueError("Unsupported momentum type {}. Must be one of {}."
                           .format(momentum_type, legal_momentum_types))
        if momentum_type != "regular" and norm_constraint is not None:
          raise ValueError("Update clipping is only supported with momentum"
                           "type 'regular'.")
        if momentum_type not in ["regular", "adam"] and momentum != 0:
          raise ValueError("Momentum must be unspecified if using a momentum_type "
                           "other than 'regular' or 'adam'.")

        self._momentum = ops.convert_to_tensor(momentum, name="momentum")
        self._momentum_type = momentum_type
        self._norm_constraint = norm_constraint

        # this is a bit of a hack
        # TODO(duckworthd): Handle this in a better way (e.g. pass it in?)
        self._batch_size = array_ops.shape(layer_collection.losses[0].inputs)[0]
        self._losses = layer_collection.losses

        self.cov_update_op = self._fisher_est.cov_update_op
        self.inv_update_op = self._fisher_est.inv_update_op
        self.inv_updates_dict = self._fisher_est.inv_updates_dict

        super(KfacOptimizer, self).__init__(learning_rate, name=name)

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

