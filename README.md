# DRLBox: Deep Reinforcement Learning as a (black) Box
Supports *only Python3* (oops).

## Designing principle
Most (deep) RL algorithms work by optimizing a neural network through interacting with a learning environment.  The goal of this package is to **minimize** the implementation effort of RL practitioners.  They only need to implement (or, more commonly, wrap) an **OpenAI-gym environment** and a neural network they want to use (as a **`tf.keras` model**) in order to run RL algorithms.

## Install
`pip install -e .` in your favorite virtual env.

## Requirements
- tensorflow==1.5.0rc0
- gym>=0.9.3
- gym[atari] (optional)

# Usage
## Asynchronous RL learner
A Python3 "binary" script called `drlbox_async.py` will be found under your (virtual env) `$PATH` after installation.  Its `--env` flag sets the user-implemented OpenAI-gym environment part, and its `--feature` flag sets the user-implemented `tf.keras` model.  Optionally, the user may specify some algorithmic configurations/hyperparameters by using the `--config` flag.  For convenience, additional paths containing the user-implemented files can be added through the `--import_path` flag.

### Use of `--env`: Implementing an OpenAI-gym environment maker
The user is supposed to implement a `make_env(*args)` function which takes in some (arbitrary number of) arguments from the command line and returns **an OpenAI-gym environment** and **the environment's name**.  Things like history stacking/frame skipping/reward engineering are usually handled here as well.

By default, the package provides a trivial example `drlbox/env/default.py`:
```python
import gym

def make_env(name):
    return gym.make(name), name
```
which takes in a name of the environment, let `gym` make that environment, and returns that environment and its name.

The `--env` flag accepts multiple arguments, and when the number of arguments is 1, it uses the default environment maker `drlbox/env/default.py` and treats the argument as the name of the environment.  When there are more than 1 arguments, the last argument will be interpreted as the filename containing the implementation of `make_env`, and all other arguments will be thrown into `make_env` (so better let them be strings).


### Use of `--feature`: Implementing (part of) a `tf.keras` model
The user is supposed to implement two functions here:

1. A `state_to_input(state)` function which takes in the `observation` from the output of the OpenAI-gym environment's `reset` or `step` function ([explanation](https://gym.openai.com/docs)) and returns something that a `tf.keras` model can directly take in.  Usually, this function does `numpy` stackings/reshapings/etc.

2. A `make_feature(observation_space, *args)` function which takes in an `observation_space` ([explanation](https://gym.openai.com/docs)) and other arguments from the command line.  It returns `inp_state`, a `tf.keras.layers.Input` layer, and `feature`, a `tf.keras` layer (when, say, actor and critic streams share a common stack of layers) or a tuple of 2 `tf.keras` layers (when actor and critic are two totally separate streams).


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
    arch_str:          Architecture of the neural net, e.g., '16 16 16'.
'''
def make_feature(observation_space, arch_str):
    net_arch = arch_str.split(' ')
    net_arch = [int(num) for num in net_arch]
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
which makes a fully-connected neural network.

The `--feature` flag also accepts multiple arguments, and when the number of arguments is 1, it uses the default neural network feature maker `drlbox/feature/fc.py` and treats the argument as the `arch_str` argument for the default maker.  When there are more than 1 arguments, the last argument will be interpreted as the filename containing the implementation the 2 required funtions, and all other arguments will be thrown into the `make_feature` function as arguments after `observation_space` (so better let them be strings as well).

### Use of `--config`:
Please refer to `drlbox/config/async_default.py`.  Generally it's something like making a `foo.py` similar to it and setting `--config foo.py`


