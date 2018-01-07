
from tensorflow.contrib.kfac.python.ops.layer_collection import LayerCollection
from tensorflow.contrib.kfac.python.ops import loss_functions as lf


class LayerCollectionWithVariance(LayerCollection):

    def register_normal_predictive_distribution_with_variance(self,
                                                              mean,
                                                              var,
                                                              seed=None,
                                                              targets=None):
        """Registers a normal predictive distribution with variable variance.
        Args:
          mean: The mean vector defining the distribution.
          var: The variance vector defining the distribution.
          seed: The seed for the RNG (for debugging) (Default: None)
          targets: (OPTIONAL) The targets for the loss function.  Only required if
            one wants to call total_loss() instead of total_sampled_loss().
            total_loss() is required, for example, to estimate the
            "empirical Fisher" (instead of the true Fisher).
            (Default: None)
        """
        loss = lf.NormalMeanVarianceNegativeLogProbLoss(
            mean, var, targets=targets, seed=seed)
        self.losses.append(loss)
