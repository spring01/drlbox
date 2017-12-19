
import gym.spaces
from tensorflow.contrib.keras import layers, models, initializers


def state_to_input(state):
    return state.ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    action_space:      Action space of the environment; Discrete;
    arch_str:          Architecture of the q net, e.g., '16 16 16'.
'''
def model(observation_space, action_space, arch_str):
    net_arch = net_arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
    state = layers.Input(shape=observation_space.spaces[-1].shape)
    feature = state
    for num_hid in net_arch:
        feature = layers.Dense(num_hid, activation='relu')(feature)
    if not isinstance(action_space, gym.spaces.discrete.Discrete):
        raise ValueError('action_space must be discrete in DQN')
    q_value = layers.Dense(action_space.n)(feature)
    return models.Model(inputs=state, outputs=q_value)

