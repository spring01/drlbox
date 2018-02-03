
from tensorflow.python.keras.layers import Input, Activation
from drlbox.layers.noisy_dense import NoisyDenseIG
from .fc import state_to_input, input_shape


'''
Input arguments:
    observation_space: Observation space of the environment;
    net_arch:          Architecture of the neural net, e.g., [16, 16, 16].
'''
def make_feature(observation_space, net_arch):
    inp_state = Input(shape=input_shape(observation_space))
    feature = inp_state
    for num_hid in net_arch:
        feature = NoisyDenseIG(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature


