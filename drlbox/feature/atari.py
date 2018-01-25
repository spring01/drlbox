
import numpy as np
from tensorflow.python.keras import layers, backend as K


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
def make_feature(observation_space, net_name='fc', net_size=512):
    net_size = int(net_size)
    net_name = net_name.lower()
    num_frames = len(observation_space.spaces)
    height, width = observation_space.spaces[0].shape
    input_shape = height, width, num_frames

    # input state
    inp_state = layers.Input(shape=input_shape)

    # convolutional layers
    conv1_32 = layers.Conv2D(32, (8, 8), strides=(4, 4))
    conv2_64 = layers.Conv2D(64, (4, 4), strides=(2, 2))
    conv3_64 = layers.Conv2D(64, (3, 3), strides=(1, 1))
    relu = layers.Activation('relu')

    # if recurrent net then change input shape
    if 'lstm' in net_name or 'gru' in net_name:
        # recurrent net
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = layers.Lambda(lambda_perm_state)(inp_state)
        dist_state = layers.Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = layers.TimeDistributed(conv1_32)(dist_state)
        dist_conv1 = layers.TimeDistributed(relu)(dist_conv1)
        dist_conv2 = layers.TimeDistributed(conv2_64)(dist_conv1)
        dist_conv2 = layers.TimeDistributed(relu)(dist_conv2)
        dist_convf = layers.TimeDistributed(conv3_64)(dist_conv2)
        dist_convf = layers.TimeDistributed(relu)(dist_convf)
        feature = layers.TimeDistributed(layers.Flatten())(dist_convf)

        # specify final hidden layer type
        if 'lstm' in net_name:
            hidden_layer = layers.LSTM
        elif 'gru' in net_name:
            hidden_layer = layers.GRU
    elif 'fc' in net_name:
        # fully connected final hidden layer
        # extract features with convolutional layers
        conv1 = conv1_32(inp_state)
        conv1 = relu(conv1)
        conv2 = conv2_64(conv1)
        conv2 = relu(conv2)
        convf = conv3_64(conv2)
        convf = relu(convf)
        feature = layers.Flatten()(convf)

        # specify final hidden layer type
        hidden_layer = layers.Dense
    else:
        raise ValueError('`net_name` is not recognized')

    # actor (policy) and critic (value) stream
    feature = hidden_layer(net_size)(feature)
    feature = relu(feature)
    return inp_state, feature

