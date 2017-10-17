
from tensorflow.contrib.keras.api.keras \
    import layers, initializers, models, backend as K
Input, Dense, Lambda = layers.Input, layers.Dense, layers.Lambda
Conv2D, Flatten = layers.Conv2D, layers.Flatten
LSTM, GRU, TimeDistributed = layers.LSTM, layers.GRU, layers.TimeDistributed

'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_name:    Name of the actor-critic net, e.g., 'fc' (fully connected);
    net_size:    Number of neurons in the first non-convolutional layer.
'''
def atari_acnet(input_shape, num_actions, net_name, net_size):
    net_name = net_name.lower()
    state, feature, net = _atari_state_feature_net(input_shape, net_name)

    # actor (policy) and critic (value) stream
    hid = net(net_size, activation='relu')(feature)
    near_zeros = initializers.RandomNormal(stddev=1e-3)
    logits = Dense(num_actions, kernel_initializer=near_zeros)(hid)
    value = Dense(1)(hid)

    # build model
    return models.Model(inputs=state, outputs=[value, logits])


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_name:    Name of the q-net, e.g., 'dqn';
    net_size:    Number of neurons in the first non-convolutional layer.
'''
def atari_qnet(input_shape, num_actions, net_name, net_size):
    net_name = net_name.lower()
    state, feature, net = _atari_state_feature_net(input_shape, net_name)

    # dueling or regular dqn/drqn
    if 'dueling' in net_name:
        value1 = net(net_size, activation='relu')(feature)
        adv1 = net(net_size, activation='relu')(feature)
        value2 = Dense(1)(value1)
        adv2 = Dense(num_actions)(adv1)
        mean_adv2 = Lambda(lambda x: K.mean(x, axis=1))(adv2)
        ones = K.ones([1, num_actions])
        lambda_exp = lambda x: K.dot(K.expand_dims(x, axis=1), -ones)
        exp_mean_adv2 = Lambda(lambda_exp)(mean_adv2)
        sum_adv = layers.add([exp_mean_adv2, adv2])
        exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = layers.add([exp_value2, sum_adv])
    else:
        hid = net(net_size, activation='relu')(feature)
        q_value = Dense(num_actions)(hid)

    # build model
    return models.Model(inputs=state, outputs=q_value)


def _atari_state_feature_net(input_shape, net_name):
    # input state
    state = Input(shape=input_shape)

    # convolutional layers
    conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'lstm' in net_name or 'gru' in net_name:
        # recurrent net
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = Lambda(lambda_perm_state)(state)
        dist_state = Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = TimeDistributed(conv1_32)(dist_state)
        dist_conv2 = TimeDistributed(conv2_64)(dist_conv1)
        dist_convf = TimeDistributed(conv3_64)(dist_conv2)
        feature = TimeDistributed(Flatten())(dist_convf)

        # specify net type for the following layer
        if 'lstm' in net_name:
            net = LSTM
        elif 'gru' in net_name:
            net = GRU
    elif 'fc' in net_name:
        # fully connected net
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = Flatten()(convf)

        # specify net type for the following layer
        net = Dense
    else:
        raise ValueError('`net_name` is not recognized')

    return state, feature, net

