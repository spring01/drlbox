
import numpy as np
from keras.layers import Input, Conv2D, Flatten, Dense, Lambda
from keras.layers import TimeDistributed, LSTM, GRU
from keras.layers import add, dot
from keras.models import Model
from keras import backend as K


def qnetwork_add_arguments(parser):
    parser.add_argument('--qnet_name', default='dqn', type=str,
        help='Q-net name')
    parser.add_argument('--qnet_size', default=256, type=int,
        help='Number of hidden units in the first non-convolutional layer')

'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    args: Need field `args.qnet_name` and `args.qnet_size`:
        args.qnet_name: Name of the q-net, e.g., 'dqn';
        args.qnet_size: Number of neurons in the first non-convolutional layer.
'''
def qnetwork(input_shape, num_actions, args):
    # input state
    state = Input(shape=input_shape)

    # convolutional layers
    conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'drqn' in args.qnet_name.lower():
        # recurrent net (drqn)
        height, width, num_frames = input_shape
        state_shape_drqn = num_frames, height, width, 1
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = Lambda(lambda_perm_state)(state)
        dist_state = Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = TimeDistributed(conv1_32)(dist_state)
        dist_conv2 = TimeDistributed(conv2_64)(dist_conv1)
        dist_convf = TimeDistributed(conv3_64)(dist_conv2)
        feature = TimeDistributed(Flatten())(dist_convf)
    elif 'dqn' in args.qnet_name.lower():
        # fully connected net (dqn)
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = Flatten()(convf)

    # network type. Dense for dqn; LSTM or GRU for drqn
    if 'lstm' in args.qnet_name.lower():
        net_type = LSTM
    elif 'gru' in args.qnet_name.lower():
        net_type = GRU
    else:
        net_type = Dense

    # dueling or regular dqn/drqn
    if 'dueling' in args.qnet_name.lower():
        value1 = net_type(args.qnet_size, activation='relu')(feature)
        adv1 = net_type(args.qnet_size, activation='relu')(feature)
        value2 = Dense(1)(value1)
        adv2 = Dense(num_actions)(adv1)
        mean_adv2 = Lambda(lambda x: K.mean(x, axis=1))(adv2)
        ones = K.ones([1, num_actions])
        lambda_exp = lambda x: K.dot(K.expand_dims(x, axis=1), -ones)
        exp_mean_adv2 = Lambda(lambda_exp)(mean_adv2)
        sum_adv = add([exp_mean_adv2, adv2])
        exp_value2 = Lambda(lambda x: K.dot(x, ones))(value2)
        q_value = add([exp_value2, sum_adv])
    else:
        hid = net_type(args.qnet_size, activation='relu')(feature)
        q_value = Dense(num_actions)(hid)

    # build model
    act = Input(shape=(num_actions,))
    q_value_act = dot([q_value, act], axes=1)
    return Model(inputs=[state, act], outputs=[q_value_act, q_value])

'''
When a state is represented as a list of frames, this interface converts it
to a correctly shaped numpy array which can be fed into the neural network
'''
def interface_list_of_frames(state):
    return np.stack(state, axis=-1).astype(np.float32)


