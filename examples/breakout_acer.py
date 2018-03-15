
import gym
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.layers import (Input, Conv2D, Activation, Lambda,
    TimeDistributed, LSTM, GRU, Flatten, Dense)
from examples.atari_wrappers import (HistoryStacker, RewardClipper,
    EpisodicLife, Preprocessor)
from drlbox.trainer import make_trainer


'''
Make a properly wrapped Atari env
'''
def make_env(name, num_frames=4, act_steps=2):
    env = gym.make(name)
    env = Preprocessor(env, shape=(84, 84))
    env = HistoryStacker(env, num_frames, act_steps)
    env = RewardClipper(env, -1.0, 1.0)
    env = EpisodicLife(env)
    return env


'''
When a state is represented by a list of frames, this interface converts it
to a correctly shaped, correctly typed numpy array which can be fed into
the convolutional neural network.
'''
def state_to_input(state):
    return np.stack(state, axis=-1).astype(np.float32)


'''
Build a convolutional actor-critic net that is similar to the Nature paper one.
Input arguments:
    env:        Atari environment.
'''
def make_model(env):
    num_frames = len(env.observation_space.spaces)
    height, width = env.observation_space.spaces[0].shape
    input_shape = height, width, num_frames

    # input state
    ph_state = Input(shape=input_shape)

    # convolutional layers
    conv1 = Conv2D(32, (8, 8), strides=(4, 4))(ph_state)
    conv1 = Activation('relu')(conv1)
    conv2 = Conv2D(64, (4, 4), strides=(2, 2))(conv1)
    conv2 = Activation('relu')(conv2)
    conv3 = Conv2D(64, (3, 3), strides=(1, 1))(conv2)
    conv3 = Activation('relu')(conv3)
    conv_flat = Flatten()(conv3)
    feature = Dense(512)(conv_flat)
    feature = Activation('relu')(feature)

    # actor (policy) and critic (value) streams
    size_logits = size_value = env.action_space.n
    logits_init = tf.keras.initializers.RandomNormal(stddev=1e-3)
    logits = Dense(size_logits, kernel_initializer=logits_init)(feature)
    value = Dense(size_value)(feature)
    return tf.keras.models.Model(inputs=ph_state, outputs=[logits, value])


'''
ACER on Breakout-v0
'''
if __name__ == '__main__':
    trainer = make_trainer('acer',
        env_maker=lambda: make_env('Breakout-v0'),
        model_maker=make_model,
        state_to_input=state_to_input,
        num_parallel=1,
        train_steps=1000,
        rollout_maxlen=4,
        batch_size=8,
        verbose=True,
        )
    trainer.run()

