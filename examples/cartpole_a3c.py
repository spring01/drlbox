
import gym
from tensorflow.python.keras.layers import Input, Dense, Activation
from drlbox.trainer import make_trainer


'''
Input arguments:
    observation_space: Observation space of the environment;
    num_hid_list:      List of hidden unit numbers in the fully-connected net.
'''
def make_feature(observation_space, num_hid_list):
    inp_state = Input(shape=observation_space.shape)
    feature = inp_state
    for num_hid in num_hid_list:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature


'''
A3C, CartPole-v0
'''
if __name__ == '__main__':
    trainer = make_trainer(
        algorithm='a3c',
        env_maker=lambda: gym.make('CartPole-v0'),
        feature_maker=lambda obs_space: make_feature(obs_space, [200, 100]),
        num_parallel=1,
        train_steps=1000,
        verbose=True,
        )
    trainer.run()
