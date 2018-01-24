
from drlbox.layers.noisy_dense import NoisyDenseIG
from drlbox.common.rlnet import RLNet
from drlbox.async.acnet import ACNet
from drlbox.dqn.qnet import QNet


class NoisyNet(RLNet):

    def __init__(self, model):
        super().__init__(model)
        self.noise_list = []
        for layer in model.layers:
            if type(layer) is NoisyDenseIG:
                self.noise_list.extend([layer.kernel_noise, layer.bias_noise])

    def sync(self):
        super().sync()
        for noise in self.noise_list:
            self.sess.run(noise.initializer)


class NoisyACNet(NoisyNet, ACNet):
    pass

class NoisyQNet(NoisyNet, QNet):
    pass

