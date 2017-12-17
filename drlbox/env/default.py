
import gym

def make_env(name='CartPole-v0'):
    return gym.make(name), name

