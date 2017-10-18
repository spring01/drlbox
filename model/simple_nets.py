
from tensorflow.contrib.keras.api.keras import layers, models, initializers
Input, Dense = layers.Input, layers.Dense
RandomNormal = initializers.RandomNormal


def state_to_input(state):
    return state[-1]

'''
Input arguments:
    observation_space: Observation space of the environment; Box;
    action_space:      Action space of the environment; Discrete;
    net_arch_str:      Architecture of the actor-critic net, e.g., '16 16 16'.
'''
def acnet(observation_space, action_space, net_arch_str):
    state, feature = _simple_state_feature(observation_space, net_arch_str)
    near_zeros = RandomNormal(stddev=1e-3)
    logits = Dense(action_space.n, kernel_initializer=near_zeros)(feature)
    value = Dense(1)(feature)
    return models.Model(inputs=state, outputs=[value, logits])


'''
Input arguments:
    observation_space: Observation space of the environment; Box;
    action_space:      Action space of the environment; Discrete;
    net_arch_str:      Architecture of the q-net, e.g., '16 16 16'.
'''
def qnet(observation_space, action_space, net_arch_str):
    state, feature = _simple_state_feature(observation_space, net_arch_str)
    q_value = Dense(action_space.n)(feature)
    return models.Model(inputs=state, outputs=q_value)


def _simple_state_feature(observation_space, net_arch_str):
    net_arch = net_arch_str.split(' ')
    state = Input(shape=observation_space.spaces[0].shape)
    feature = state
    for num_hid in net_arch:
        feature = Dense(num_hid, activation='relu')(feature)
    return state, feature


Preprocessor = None

