
from drlbox.layers.noisy_dense import NoisyDenseIG
from .rlnet import RLNet
from .acnet import ACNet
from .qnet import QNet


class NoisyNet(RLNet):

    dense_layer = NoisyDenseIG

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.noise_list = []
        for layer in self.model.layers:
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

