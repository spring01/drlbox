
from tensorflow import keras
from drlbox.layers.preact_layers import DensePreact


def state_to_input(state):
    return state.ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    arch_str:          Architecture of the neural net, e.g., '16 16 16'.
'''
def make_feature(observation_space, arch_str):
    net_arch = arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
    inp_state = keras.layers.Input(shape=observation_space.shape)
    feature1 = inp_state
    for num_hid in net_arch:
        feature1 = DensePreact(num_hid, activation='relu')(feature1)
    feature2 = inp_state
    for num_hid in net_arch:
        feature2 = DensePreact(num_hid, activation='relu')(feature2)
    return inp_state, (feature1, feature2)


