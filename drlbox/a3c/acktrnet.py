
from .acnet import ACNet


class ACKTRNet(ACNet):

    def set_optimizer(self, optimizer, train_weights=None):
        grads_and_vars = optimizer.compute_gradients(self.tf_loss, self.weights)
        g_op = optimizer.apply_gradients(grads_and_vars, train_weights)
        self.op_train = [g_op, optimizer.cov_update_op, optimizer.inv_update_op]

