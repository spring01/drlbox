
import numpy as np
import gym.spaces
from tensorflow.python.keras.layers import Input, Dense, Activation


def state_to_input(state):
    return np.array(state).ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    arch_str:          Architecture of the neural net, e.g., '16 16 16'.
'''
def make_feature(observation_space, arch_str):
    net_arch = arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
    inp_state = Input(shape=input_shape(observation_space))
    feature = inp_state
    for num_hid in net_arch:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature


def input_shape(observation_space):
    if type(observation_space) is gym.spaces.Box:
        input_dim = np.prod(observation_space.shape)
    elif type(observation_space) is gym.spaces.Tuple:
        input_dim = sum(np.prod(sp.shape) for sp in observation_space.spaces)
    else:
        raise TypeError('Type of observation_space is not recognized')
    return input_dim,


