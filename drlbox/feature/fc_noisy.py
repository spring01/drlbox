
from tensorflow import keras
from drlbox.layers.noisy_dense import NoisyDenseIG


def state_to_input(state):
    return state.ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    arch_str:          Architecture of the actor-critic net, e.g., '16 16 16'.
'''
def feature(observation_space, arch_str):
    net_arch = arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
    state = keras.layers.Input(shape=observation_space.shape)
    feature = state
    for num_hid in net_arch:
        feature = NoisyDenseIG(num_hid, activation='relu')(feature)
    return state, feature


