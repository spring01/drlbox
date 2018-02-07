
from drlbox.layer.noisy_dense import NoisyDenseIG
from .net_base import RLNet
from .ac_net import ACNet
from .acer_net import ACERNet
from .q_net import QNet


class NoisyNet(RLNet):

    dense_layer = NoisyDenseIG

    def set_model(self, model):
        super().set_model(model)
        self.noise_list = []
        for layer in model.layers:
            if type(layer) is NoisyDenseIG:
                self.noise_list.extend([layer.kernel_noise, layer.bias_noise])

    def sync(self):
        super().sync()
        for noise in self.noise_list:
            self.sess.run(noise.initializer)

    @staticmethod
    def load_model(filename):
        return RLNet.load_model(filename, {'NoisyDenseIG': NoisyDenseIG})


class NoisyACNet(NoisyNet, ACNet):
    pass

class NoisyACERNet(NoisyNet, ACERNet):
    pass

class NoisyQNet(NoisyNet, QNet):
    pass

