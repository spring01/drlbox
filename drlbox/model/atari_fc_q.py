
from tensorflow.contrib.keras import layers, models, backend as K
from .atari_fc_ac import atari_state_feature_net

import numpy as np
'''
When a state is represented as a list of frames, this interface converts it
to a correctly shaped numpy array which can be fed into the neural network
'''
def state_to_input(state):
    return np.stack(state, axis=-1).astype(np.float32)


'''
Input arguments:
    observation_space: Observation space of the environment;
    action_space:      Action space of the environment; Discrete;
    net_name:          Name of the q net, e.g., 'fc';
    net_size:          Number of neurons in the first non-convolutional layer.
'''
def model(observation_space, action_space, net_name='fc', net_size=512):
    num_actions = action_space.n
    net_size = int(net_size)
    net_name = net_name.lower()
    state, feature, net = atari_state_feature_net(observation_space, net_name)

    # dueling or regular dqn/drqn
    if 'dueling' in net_name:
        value1 = net(net_size, activation='relu')(feature)
        adv1 = net(net_size, activation='relu')(feature)
        value2 = layers.Dense(1)(value1)
        adv2 = layers.Dense(num_actions)(adv1)
        mean_adv2 = layers.Lambda(lambda x: K.mean(x, axis=1))(adv2)
        ones = K.ones([1, num_actions])
        lambda_exp = lambda x: K.dot(K.expand_dims(x, axis=1), -ones)
        exp_mean_adv2 = layers.Lambda(lambda_exp)(mean_adv2)
        sum_adv = layers.add([exp_mean_adv2, adv2])
        exp_value2 = layers.Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = layers.add([exp_value2, sum_adv])
    else:
        hid = net(net_size, activation='relu')(feature)
        q_value = layers.Dense(num_actions)(hid)

    # build model
    return models.Model(inputs=state, outputs=q_value)

