# DRLBox: Deep Reinforcement Learning as a (black) Box
Supports *only Python3* (oops).

## Designing principle
Most (deep) RL algorithms work by optimizing a neural network through interacting with a learning environment.  The goal of this package is to **minimize** the implementation effort of RL practitioners.  They only need to implement (or, more commonly, wrap) an **OpenAI-gym environment** and a neural network they want to use as a **`tf.keras` model** (along with an interface function that turns the `observation` from the gym environment into a format that can be fed into the `tf.keras` model), in order to run RL algorithms.

## Install
`pip install -e .` in your favorite virtual env.

## Requirements
- tensorflow>=1.5.0
- gym>=0.9.6
- gym[atari] (optional)

## Currently implemented RL algorithms
- **Actor-critic series**
  - **A3C**  (https://arxiv.org/abs/1602.01783)  Actor-critic, using a critic-based advantage function as the baseline for variance reduction, asynchronous parallel.
  - **ACKTR**  (https://arxiv.org/abs/1708.05144)  A3C with the K-FAC optimizer instead of Adam.  This implementation is an asynchronous variant of the original version based on A2C, and so weight updates will be more frequent but the K-FAC curvature estimate may be less accurate.  Relies on `tf.contrib.kfac` and so currently the neural net may only contain `Dense` and `Conv2D` layers.
  - **ACER**  (https://arxiv.org/abs/1611.01224)  A3C with replay for off-policy learning.  The critic becomes a state-action value function instead of a state-only function.  The authors proposed a trust-region optimization scheme based on the KL divergence wrt a Polyak averaging policy network.  This implementation however includes the KL divergence (with a tunable scale factor) in the total loss.  This choice is less stable wrt change in hyperparameters, but simplifies the combination of ACER and ACKTR.
  - **ACERKTR**  (orz)  ACER with the K-FAC optimizer instead of Adam.
  - **NoisyNetA3C** and **NoisyNetACER**  (https://arxiv.org/abs/1706.10295)  Introduces (independent) Gaussian noises to policy network weights.  Allows the exploration strategy to change across different training stages and adapt to different parts of the state representation.

- **DQN series**
  - **DQN**  (https://arxiv.org/abs/1602.01783)  Asynchronous multi-step Q-learning without replay memory.
  - **NoisyNetDQN**  (https://arxiv.org/abs/1706.10295)  DQN + NoisyNet.  Uses independent Gaussian noises as opposed to the factored Gaussian noises used in the original paper.

# Usage
A minimal demo could be as simple as the following code snippet.  (A3C algorithm, `CartPole-v0` environment, and a 1-layer fully-connected net with 100 hidden units)
```
from drlbox.env.default import make_env
from drlbox.feature.fc import state_to_input, make_feature
from drlbox.trainer import make_trainer


trainer = make_trainer(algorithm='a3c',
                       env_maker=lambda: make_env('CartPole-v0'),
                       feature_maker=lambda o: make_feature(o, [100]),
                       state_to_input=state_to_input,
                       num_parallel=1,
                       train_steps=1000,)
trainer.run()
```

## Gym Environment
### Implementing an OpenAI-gym environment maker
The user is supposed to implement a `env_maker` callable which returns **an OpenAI-gym environment**.  Things like history stacking/frame skipping/reward engineering are usually handled here as well.

By default, the package provides a trivial example `drlbox/env/default.py`:
```python
import gym

def make_env(name):
    return gym.make(name)
```
which takes in a name of the environment, let `gym` make the environment, and returns the environment.  To use the default env maker, simply let the callable be `env_maker = lambda: make_env(ENV_NAME)`.


## Neural network
### Implementing (part of) a `tf.keras` model
The user is supposed to implement a `feature_maker` callable which takes in an `observation_space` ([explanation](https://gym.openai.com/docs)) and returns `inp_state`, a `tf.keras.layers.Input` layer, and `feature`, a `tf.keras` layer or a tuple of 2 `tf.keras` layers.  For example, with actor-critic algorithms, when `feature` is a `tf.keras` layer, the actor and the critic streams share a common stack of layers. When `feature` is a tuple of 2 `tf.keras` layers, the actor and the critic will be completely separated).

### Implementing an interface function
The user is also supposed to implement a `state_to_input` callable which takes in the `observation` from the output of the OpenAI-gym environment's `reset` or `step` function ([explanation](https://gym.openai.com/docs)) and returns something that a `tf.keras` model can directly take in.  Usually, this function does stuffs like `numpy` stackings/reshapings/etc.

### Example
By default, the package provides a trivial example `drlbox/feature/fc.py`:
```python
import numpy as np
import gym.spaces
from tensorflow.python.keras.layers import Input, Dense, Activation


def state_to_input(state):
    return np.array(state).ravel()

'''
Input arguments:
    observation_space: Observation space of the environment;
    net_arch:          Architecture of the neural net, e.g., [16, 16, 16].
'''
def make_feature(observation_space, net_arch):
    inp_state = Input(shape=input_shape(observation_space))
    feature = inp_state
    for num_hid in net_arch:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    return inp_state, feature


def input_shape(observation_space):
    if type(observation_space) is gym.spaces.Box:
        input_dim = np.prod(observation_space.shape)
    elif type(observation_space) is gym.spaces.Tuple:
        input_dim = sum(np.prod(sp.shape) for sp in observation_space.spaces)
    else:
        raise TypeError('Type of observation_space is not recognized')
    return input_dim,
```
which makes a fully-connected neural network until the last layer before the value/policy layer.  To use the default feature maker, simply let the feature-maker callable be `feature_maker = lambda o: make_feature(o, ARCHITECTURE)`.  The function `state_to_input` can be directly used as the default interface function.

