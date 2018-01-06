
import gym.spaces
from tensorflow.python.keras import layers, models, initializers


'''
Input arguments:
    state:          Model input;
    feature:        Output of the feature function;
    action_space:   Action space of the environment;
'''
def actor_critic_model(state, feature, action_space):
    if isinstance(feature, tuple):
        # totally splitted logits/value streams when feature is a length 2 tuple
        feature_logits, feature_value = feature
    else:
        # feature is a single stream otherwise
        feature_logits = feature_value = feature
    if isinstance(action_space, gym.spaces.discrete.Discrete): # discrete action
        size_logits = action_space.n
        action_mode = 'discrete'
        init = initializers.RandomNormal(stddev=1e-3)
    elif isinstance(action_space, gym.spaces.box.Box): # continuous action
        size_logits = len(action_space.shape) + 1
        action_mode = 'continuous'
        init = 'glorot_uniform'
    else:
        raise ValueError('type of action_space is illegal')
    logits = layers.Dense(size_logits, kernel_initializer=init)(feature_logits)
    value = layers.Dense(1)(feature_value)
    model = models.Model(inputs=state, outputs=[value, logits])
    model.action_mode = action_mode
    return model


