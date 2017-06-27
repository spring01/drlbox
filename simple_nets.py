
import keras.layers as kl
from keras.initializers import RandomNormal
from keras.models import Model


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the actor-critic net.
'''
def simple_acnet(input_shape, num_actions, net_arch):
    state, feature = _simple_state_feature(input_shape, net_arch)
    near_zeros = RandomNormal(stddev=1e-3)
    logits = kl.Dense(num_actions, kernel_initializer=near_zeros)(feature)
    value = kl.Dense(1)(feature)
    return Model(inputs=state, outputs=[value, logits])


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the actor-critic net.
'''
def simple_qnet(input_shape, num_actions, net_arch):
    state, feature = _simple_state_feature(input_shape, net_arch)
    q_value = kl.Dense(num_actions)(feature)
    value = kl.Dense(1)(feature)
    return Model(inputs=state, outputs=q_value)


def _simple_state_feature(input_shape, net_arch):
    state = kl.Input(shape=input_shape)
    feature = state
    for num_hid in net_arch:
        feature = kl.Dense(num_hid, activation='relu')(feature)
    return state, feature


