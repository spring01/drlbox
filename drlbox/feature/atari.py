
import numpy as np
from tensorflow.python.keras import layers, backend as K
from drlbox.layers.preact_layers import DensePreact, Conv2DPreact

'''
When a state is represented as a list of frames, this interface converts it
to a correctly shaped numpy array which can be fed into the neural network
'''
def state_to_input(state):
    return np.stack(state, axis=-1).astype(np.float32)


'''
Input arguments:
    observation_space: Observation space of the environment; Tuple of Boxes;
    net_name:          Name of the neural net, e.g., 'fc';
    net_size:          Number of neurons in the first non-convolutional layer.
'''
def feature(observation_space, net_name='fc', net_size=512):
    net_size = int(net_size)
    net_name = net_name.lower()
    num_frames = len(observation_space.spaces)
    height, width = observation_space.spaces[0].shape
    input_shape = height, width, num_frames

    # input state
    state = layers.Input(shape=input_shape)

    # convolutional layers
    conv1_32 = Conv2DPreact(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2DPreact(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2DPreact(64, (3, 3), strides=(1, 1), activation='relu')

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

        # specify final hidden layer type
        if 'lstm' in net_name:
            hidden_layer = layers.LSTM
        elif 'gru' in net_name:
            hidden_layer = layers.GRU
    elif 'fc' in net_name:
        # fully connected final hidden layer
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = layers.Flatten()(convf)

        # specify final hidden layer type
        hidden_layer = DensePreact
    else:
        raise ValueError('`net_name` is not recognized')

    # actor (policy) and critic (value) stream
    feature = hidden_layer(net_size, activation='relu')(feature)
    return state, feature

