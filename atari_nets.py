
import tensorflow.contrib.keras.api.keras.layers as kl
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.initializers import RandomNormal
from tensorflow.contrib.keras.api.keras.models import Model


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_name:    Name of the actor-critic net, e.g., 'fully connected';
    net_size:    Number of neurons in the first non-convolutional layer.
'''
def atari_acnet(input_shape, num_actions, net_name, net_size):
    net_name = net_name.lower()
    state, feature, net = _atari_state_feature_net(input_shape, net_name)

    # actor (policy) and critic (value) stream
    hid = net(net_size, activation='relu')(feature)
    near_zeros = RandomNormal(stddev=1e-3)
    logits = kl.Dense(num_actions, kernel_initializer=near_zeros)(hid)
    value = kl.Dense(1)(hid)

    # build model
    return Model(inputs=state, outputs=[value, logits])


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
        value2 = kl.Dense(1)(value1)
        adv2 = kl.Dense(num_actions)(adv1)
        mean_adv2 = kl.Lambda(lambda x: K.mean(x, axis=1))(adv2)
        ones = K.ones([1, num_actions])
        lambda_exp = lambda x: K.dot(K.expand_dims(x, axis=1), -ones)
        exp_mean_adv2 = kl.Lambda(lambda_exp)(mean_adv2)
        sum_adv = kl.add([exp_mean_adv2, adv2])
        exp_value2 = kl.Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = kl.add([exp_value2, sum_adv])
    else:
        hid = net(net_size, activation='relu')(feature)
        q_value = kl.Dense(num_actions)(hid)

    # build model
    return Model(inputs=state, outputs=q_value)


def _atari_state_feature_net(input_shape, net_name):
    # input state
    state = kl.Input(shape=input_shape)

    # convolutional layers
    conv1_32 = kl.Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = kl.Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = kl.Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'lstm' in net_name or 'gru' in net_name:
        # recurrent net
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = kl.Lambda(lambda_perm_state)(state)
        dist_state = kl.Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = kl.TimeDistributed(conv1_32)(dist_state)
        dist_conv2 = kl.TimeDistributed(conv2_64)(dist_conv1)
        dist_convf = kl.TimeDistributed(conv3_64)(dist_conv2)
        feature = kl.TimeDistributed(kl.Flatten())(dist_convf)

        # specify net type for the following layer
        if 'lstm' in net_name:
            net = kl.LSTM
        elif 'gru' in net_name:
            net = kl.GRU
    elif 'fully connected' in net_name:
        # fully connected net
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = kl.Flatten()(convf)

        # specify net type for the following layer
        net = kl.Dense

    return state, feature, net

