
from tensorflow.contrib.keras import layers, initializers, models, backend as K

import numpy as np
'''
When a state is represented as a list of frames, this interface converts it
to a correctly shaped numpy array which can be fed into the neural network
'''
def state_to_input(state):
    return np.stack(state, axis=-1).astype(np.float32)


'''
Input arguments:
    observation_space: Observation space of the environment; Tuple of Boxes;
    action_space:      Action space of the environment; Discrete;
    net_name:          Name of the actor-critic net, e.g., 'fc';
    net_size:          Number of neurons in the first non-convolutional layer.
'''
def model(observation_space, action_space, net_name='fc', net_size=512):
    num_actions = action_space.n
    net_size = int(net_size)
    net_name = net_name.lower()
    state, feature, net = atari_state_feature_net(observation_space, net_name)

    # actor (policy) and critic (value) stream
    hid = net(net_size, activation='relu')(feature)
    near_zeros = initializers.RandomNormal(stddev=1e-3)
    logits = layers.Dense(num_actions, kernel_initializer=near_zeros)(hid)
    value = layers.Dense(1)(hid)

    # build model
    model = models.Model(inputs=state, outputs=[value, logits])
    model.action_mode = 'discrete'
    return model


def atari_state_feature_net(observation_space, net_name):
    num_frames = len(observation_space.spaces)
    height, width = observation_space.spaces[0].shape
    input_shape = height, width, num_frames

    # input state
    state = layers.Input(shape=input_shape)

    # convolutional layers
    conv1_32 = layers.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = layers.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = layers.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'lstm' in net_name or 'gru' in net_name:
        # recurrent net
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = layers.Lambda(lambda_perm_state)(state)
        dist_state = layers.Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = layers.TimeDistributed(conv1_32)(dist_state)
        dist_conv2 = layers.TimeDistributed(conv2_64)(dist_conv1)
        dist_convf = layers.TimeDistributed(conv3_64)(dist_conv2)
        feature = layers.TimeDistributed(Flatten())(dist_convf)

        # specify net type for the following layer
        if 'lstm' in net_name:
            net = layers.LSTM
        elif 'gru' in net_name:
            net = layers.GRU
    elif 'fc' in net_name:
        # fully connected net
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = layers.Flatten()(convf)

        # specify net type for the following layer
        net = layers.Dense
    else:
        raise ValueError('`net_name` is not recognized')

    return state, feature, net


