
import gym.spaces
from tensorflow.contrib.keras import layers, models, initializers


'''
Input arguments:
    state:          Model input
    feature:        Output of the feature function;
    action_space:   Action space of the environment;
'''
def actor_critic_model(state, feature, action_space):
    if isinstance(action_space, gym.spaces.discrete.Discrete): # discrete action
        size_logits = action_space.n
        action_mode = 'discrete'
    elif isinstance(action_space, gym.spaces.box.Box): # continuous action
        size_logits = len(action_space.shape) + 1
        action_mode = 'continuous'
    else:
        raise ValueError('type of action_space is illegal')
    near_zeros = initializers.RandomNormal(stddev=1e-3)
    logits = layers.Dense(size_logits, kernel_initializer=near_zeros)(feature)
    value = layers.Dense(1)(feature)
    model = models.Model(inputs=state, outputs=[value, logits])
    model.action_mode = action_mode
    return model


