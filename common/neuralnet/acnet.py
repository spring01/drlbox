

import tensorflow as tf
from tensorflow.contrib.keras.python.keras.layers import \
    Input, Flatten, Lambda, Conv2D, Dense, LSTM, GRU, add, dot, TimeDistributed
from tensorflow.contrib.keras.python.keras.models import Model
from tensorflow.contrib.keras import backend as K
from .rlnet import RLNet


class ACNet(RLNet):

    def __init__(self, model):
        self.weights = model.weights
        self.ph_state, = model.inputs
        tf_value, self.tf_logits = model.outputs
        self.tf_value = tf_value[:, 0]

    def set_loss(self, entropy_weight=0.01):
        tf_logits = self.tf_logits
        log_probs = tf.nn.log_softmax(tf_logits)
        probs = tf.nn.softmax(tf_logits)

        ph_advantage = tf.placeholder(tf.float32, [None])
        ph_target = tf.placeholder(tf.float32, [None])
        ph_action = tf.placeholder(tf.float32, tf_logits.shape.as_list())

        log_probs_act = tf.reduce_sum(log_probs * ph_action, axis=1)
        policy_loss = -tf.reduce_sum(log_probs_act * ph_advantage)
        value_loss = tf.nn.l2_loss(self.tf_value - ph_target)
        entropy = -tf.reduce_sum(probs * log_probs)
        self.tf_loss = policy_loss + value_loss - entropy * entropy_weight
        self.ph_advantage = ph_advantage
        self.ph_target = ph_target
        self.ph_action = ph_action

    def action_values(self, state):
        return self.sess.run(self.tf_logits, feed_dict={self.ph_state: state})

    def value(self, state):
        return self.sess.run(self.tf_value, feed_dict={self.ph_state: state})

    def train_on_batch(self, state, action, advantage, target):
        feed_dict = {self.ph_state:     state,
                     self.ph_action:    action,
                     self.ph_advantage: advantage,
                     self.ph_target:    target}
        self.sess.run(self.op_train, feed_dict=feed_dict)


'''
Input arguments:
    input_shape: Tuple of the format (height, width, num_frames);
    num_actions: Number of actions in the environment; integer;
    acnet_name:  Name of the actor-critic net, e.g., 'fully connected';
    acnet_size:  Number of neurons in the first non-convolutional layer.
'''
def atari_acnet(input_shape, num_actions, acnet_name, acnet_size):
    acnet_name = acnet_name.lower()

    # input state
    state = Input(shape=input_shape)

    # convolutional layers
    conv1_32 = Conv2D(32, (8, 8), strides=(4, 4), activation='relu')
    conv2_64 = Conv2D(64, (4, 4), strides=(2, 2), activation='relu')
    conv3_64 = Conv2D(64, (3, 3), strides=(1, 1), activation='relu')

    # if recurrent net then change input shape
    if 'lstm' in acnet_name or 'gru' in acnet_name:
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
        if 'lstm' in acnet_name:
            net_type = LSTM
        elif 'gru' in acnet_name:
            net_type = GRU
    elif 'fully connected' in acnet_name:
        # fully connected net
        # extract features with convolutional layers
        conv1 = conv1_32(state)
        conv2 = conv2_64(conv1)
        convf = conv3_64(conv2)
        feature = Flatten()(convf)

        # specify net type for the following layer
        net_type = Dense

    # actor (policy) and critic (value) stream
    hid = net_type(acnet_size, activation='relu')(feature)
    logits = Dense(num_actions)(hid)
    value = Dense(1)(hid)

    # build model
    return Model(inputs=state, outputs=[value, logits])

