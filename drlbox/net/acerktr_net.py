
from .acer_net import ACERNet
from .acktr_net import ACKTRNet

class ACERKTRNet(ACERNet, ACKTRNet):

    def train_on_batch(self, *args, **kwargs):
        loss = super().train_on_batch(*args, **kwargs)
        self.train_step_counter += 1
        if self.train_step_counter >= self.inv_upd_interval:
            self.sess.run(self.op_inv_update)
            self.train_step_counter = 0
        return loss
