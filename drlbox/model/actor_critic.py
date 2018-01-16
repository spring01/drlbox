
import gym.spaces
from tensorflow.python.keras import layers, models, initializers
from drlbox.layers.noisy_dense import NoisyDenseIG
from drlbox.common.manager import DISCRETE, CONTINUOUS


'''
Input arguments:
    state:          Model input;
    feature:        Output of the feature function;
    action_space:   Action space of the environment;
'''
def actor_critic_model(state, feature, action_space, noisy=False):
    if type(feature) is tuple:
        # totally splitted logits/value streams when feature is a length 2 tuple
        feature_logits, feature_value = feature
    else:
        # feature is a single stream otherwise
        feature_logits = feature_value = feature
    if type(action_space) is gym.spaces.discrete.Discrete: # discrete action
        size_logits = action_space.n
        init = initializers.RandomNormal(stddev=1e-3)
    elif type(action_space) is gym.spaces.box.Box: # continuous action
        size_logits = len(action_space.shape) + 1
        init = 'glorot_uniform'
    else:
        raise ValueError('type of action_space is illegal')

    # in NoisyNet actor-critic, only the policy network is noisy
    if noisy:
        dense = NoisyDenseIG
    else:
        dense = layers.Dense
    logits = dense(size_logits, kernel_initializer=init)(feature_logits)
    value = layers.Dense(1)(feature_value)
    return models.Model(inputs=state, outputs=[value, logits])


