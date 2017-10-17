
from tensorflow.contrib.keras.api.keras import layers, models, initializers
Input, Dense = layers.Dense, layers.Input
RandomNormal = initializers.RandomNormal

'''
Input arguments:
    input_shape: Tuple of the format (dim_input,);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the actor-critic net.
'''
def simple_acnet(input_shape, num_actions, net_arch):
    state, feature = _simple_state_feature(input_shape, net_arch)
    near_zeros = RandomNormal(stddev=1e-3)
    logits = Dense(num_actions, kernel_initializer=near_zeros)(feature)
    value = Dense(1)(feature)
    return models.Model(inputs=state, outputs=[value, logits])


'''
Input arguments:
    input_shape: Tuple of the format (dim_input,);
    num_actions: Number of actions in the environment; integer;
    net_arch:    Architecture of the q-net.
'''
def simple_qnet(input_shape, num_actions, net_arch):
    state, feature = _simple_state_feature(input_shape, net_arch)
    q_value = Dense(num_actions)(feature)
    return models.Model(inputs=state, outputs=q_value)


def _simple_state_feature(input_shape, net_arch):
    state = Input(shape=input_shape)
    feature = state
    for num_hid in net_arch:
        feature = Dense(num_hid, activation='relu')(feature)
    return state, feature


