
import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import \
    Input, Flatten, Lambda, Conv2D, Dense, LSTM, GRU, add, dot, TimeDistributed
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras import backend as K
from .rlnet import RLNet


class QNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        self.tf_values, = model.outputs

    def set_loss(self, loss_function):
        tf_values = self.tf_values
        target_shape = tf_values.shape.as_list()
        ph_target = tf.placeholder(tf.float32, target_shape)
        batch_size, num_actions = target_shape
        ph_weight = tf.placeholder(tf.float32, [batch_size])
        weight_tile = tf.tile(tf.expand_dims(ph_weight, 1), [1, num_actions])
        weighted_ph_target = weight_tile * ph_target
        weighted_tf_values = weight_tile * tf_values
        self.tf_loss = loss_function(weighted_ph_target, weighted_tf_values)
        self.ph_target = ph_target
        self.ph_sample_weight = ph_weight

    def action_values(self, state):
        return self.sess.run(self.tf_values, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, target, sample_weight=None):
        if sample_weight is None:
            sample_weight = [1.0] * len(state)
        feed_dict = {self.ph_state:         state,
                     self.ph_target:        target,
                     self.ph_sample_weight: sample_weight}
        self.sess.run(self.op_train, feed_dict=feed_dict)


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    net_name:    Name of the q-net, e.g., 'dqn';
    net_size:    Number of neurons in the first non-convolutional layer.
'''
def atari_qnet(input_shape, num_actions, net_name, net_size):
    net_name = net_name.lower()

    # input state
    state = Input(shape=input_shape)

    # convolutional layers
    conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'drqn' in net_name:
        # recurrent net (drqn)
        lambda_perm_state = lambda x: K.permute_dimensions(x, [0, 3, 1, 2])
        perm_state = Lambda(lambda_perm_state)(state)
        dist_state = Lambda(lambda x: K.stack([x], axis=4))(perm_state)

        # extract features with `TimeDistributed` wrapped convolutional layers
        dist_conv1 = TimeDistributed(conv1_32)(dist_state)
        dist_conv2 = TimeDistributed(conv2_64)(dist_conv1)
        dist_convf = TimeDistributed(conv3_64)(dist_conv2)
        feature = TimeDistributed(Flatten())(dist_convf)
    elif 'dqn' in net_name:
        # fully connected net (dqn)
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = Flatten()(convf)

    # network type. Dense for dqn; LSTM or GRU for drqn
    if 'lstm' in net_name:
        net_type = LSTM
    elif 'gru' in net_name:
        net_type = GRU
    else:
        net_type = Dense

    # dueling or regular dqn/drqn
    if 'dueling' in net_name:
        value1 = net_type(net_size, activation='relu')(feature)
        adv1 = net_type(net_size, activation='relu')(feature)
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
        hid = net_type(net_size, activation='relu')(feature)
        q_value = Dense(num_actions)(hid)

    # build model
    return Model(inputs=state, outputs=q_value)

