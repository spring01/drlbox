# DRLBox: Deep Reinforcement Learning as a (black) Box
Supports *only Python3* (oops).

## Designing principle
Most (deep) RL algorithms work by optimizing a neural network through interacting with a learning environment.  The goal of this package is to **minimize** the implementation effort of RL practitioners.  They only need to implement (or, more commonly, wrap) an **OpenAI-gym environment** and a neural network they want to use as a **`tf.keras` model** (along with an interface function that turns the `observation` from the gym environment into a format that can be fed into the `tf.keras` model), in order to run RL algorithms.

## Install
`pip install -e .` in your favorite virtual env.

## Requirements
- tensorflow>=1.5.0 (lower versions may be used if `optimizer='kfac'` is never invoked)
- gym>=0.9.6
- gym[atari] (optional)

## Currently implemented RL algorithms
- **Actor-critic family**
  - **A3C**  (https://arxiv.org/abs/1602.01783)  Actor-critic, using a critic-based advantage function as the baseline for variance reduction, asynchronous parallel.
  - **ACER**  (https://arxiv.org/abs/1611.01224)  A3C with uniform replay, using the Retrace off-policy correction.  The critic becomes a state-action value function instead of a state-only function.  The authors proposed a trust-region optimization scheme based on the KL divergence wrt a Polyak averaging policy network.  This implementation however includes the KL divergence (with a tunable scale factor) in the total loss.  This choice is less stable wrt change in hyperparameters, but simplifies the combination of ACER and ACKTR.
  - **IMPALA**  (https://arxiv.org/abs/1802.01561)  A3C with replay and another (actually, a simpler) flavor of off-policy correction called V-trace.  This implementation is a lot more naive compared with the original distributed framework, however it gives an idea of how the off-policy correction is done and is much easier to integrate with ACKTR.

- **DQN family**
  - **DQN**  (https://arxiv.org/abs/1602.01783)  Asynchronous multi-step Q-learning without replay memory.

- **Algorithm related options**
  - `optimizer='kfac'`: Based on the idea of **ACKTR** (https://arxiv.org/abs/1708.05144), which is simply A3C with the K-FAC optimizer instead of Adam.  This implementation is an asynchronous variant of the original version based on synchronous algorithms (e.g., A2C), and so weight updates will be more frequent but the K-FAC curvature estimate may be less accurate.  Relies on `tf.contrib.kfac` and so currently the neural net may only contain `Dense`, `Conv2D`, and (self-implemented) `drlbox.layer.noisy_dense.NoisyNetFG` layers.
  - `noisynet='ig'` or `noisynet='fg'`: Based on the idea of **NoisyNet** (https://arxiv.org/abs/1706.10295), which introduces independent (`'ig'`) or factorized (`'fg'`) Gaussian noises to network weights.  Allows the exploration strategy to change across different training stages and adapt to different parts of the state representation.

Side note: options `noisynet='ig'` and `optimizer='kfac'` are currently not compatible with each other, as we haven't coded the K-FAC approximation for independent Gaussian noise NoisyNet layer yet.  On the other hand, `noisynet='fg'` works fine with `optimizer='kfac'`.

# Usage
## Arguments
- **Arguments shared by `trainer` and `evaluator` classes**
  - `env_maker`: *callable*.  Returns a gym env on calling.  Detailed in the **Gym Environment** section below.  Default: `None`.
  - `state_to_input`: *callable*.  Converts the `observation` from a gym env to some data (usually NumPy array) that can be fed into a `tf.keras` model.  Detailed in the **Neural network** section below.  Default: `None` (will set `self.state_to_input = lambda x: x` internally if set to `None`).
  - `load_model`: *`str`*.  File name (full path) of a `h5py` file that contains a saved `tf.keras` model (usually saved through [`tf.keras.models.Model:save`](https://www.tensorflow.org/api_docs/python/tf/keras/Model#save)).  If specified, training or evaluation will start from this model.  Default: `None`.
  - `load_model_custom`: *`dict`*.  As same as the `custom_objects` argument in [`tf.keras.models.load_model`](https://www.tensorflow.org/api_docs/python/tf/keras/models/load_model).  Default: `None`.
  - `verbose`: *`bool`*.  Whether or not to print training/evaluating information.  Default: `False`.
- **`trainer` classes common arguments**
  - `feature_maker`: *callable*.  Takes in `env.observation_space` and returns `(inp_state, feature)`, a 2-tuple of a `tf.keras.layers.Input` layer and an arbitrary typed (e.g., `tf.keras.layes.Dense`) `tf.keras` layer.  Detailed in the **Neural network** section below.  Default: `None`.
  - `model_maker`: *callable*.  Takes in a gym env and returns a `tf.keras` model.  Detailed in the **Neural network** section below.  The trainer will ignore `feature_maker` if `model_maker` is set.  Default: `None`.
  - `num_parallel`: *`int`*.  Number of parallel processes in training.  Default: number of cpu (logical) core counts.
  - `port_begin`: *`int`*.  Starting gRPC port number used by distributed tensorflow.  Default: `2220`.
  - `discount`: *`float`*.  Discount factor (gamma) in reinforcement learning.  Default: `0.99`.
  - `train_steps`: *`int`*.  Maximum number of gym env steps in training.  Default: `1000000`.
  - `rollout_maxlen`: *`int`*.  Maximum length of a rollout.  Also the number of env steps in a rollout list.  Please refer to the comments in [drlbox/trainer/trainer_base.py](drlbox/trainer/trainer_base.py) for detail explanation.  Default: `32`.
  - `batch_size`: *`int`*.  Number of rollout lists in a batch.  Please refer to the comments in [drlbox/trainer/trainer_base.py](drlbox/trainer/trainer_base.py) for details.  Default: `1`.
  - `online_learning`: *`bool`*.  Whether or not to perform online learning on a newly collected batch.  Default: `True`.
  - `replay_type`: *`None` or `str`*.  Type of the replay memory.  Choices are `[None, 'uniform']` where `None` means no replay memory.  Default: `None` (note: some algorithms such as ACER and IMPALA will set `replay_type='uniform'` by default).
  - `replay_ratio`: *`int`*.  After putting a newly collected online batch into the replay memory, a random integer number of offline, off-policy batch learnings will be performed, and the random integer number will be coming from a Poisson distribution using this argument as the Poisson parameter.  Default: `4`.
  - `replay_kwargs`: *`dict`*.  Keyword arguments that will be passed to the replay constructor after combining with the default replay keyword arguments `dict(maxlen=1000, minlen=100)`.  Default: `{}`.
  - `optimizer`: *`str` or a `tf.train.Optimizer` instance*.  `str` choices are `['adam', 'kfac']`.  Default: `adam`.
  - `opt_clip_norm`: *`float`*.  Maximum global gradient norm for gradient clipping.  Default: `40.0`.
  - `opt_kwargs`: *`dict`*.  Keyword arguments that will be passed to the optimizer constructor after combining with the default keyword arguments.  For `'adam'`, the default keyword arguments are `dict(learning_rate=1e-4, epsilon=1e-4)`.  For `'kfac'`, the default keyword arguments are `dict(learning_rate=1e-4, cov_ema_decay=0.95, damping=1e-3, norm_constraint=1e-3, momentum=0.0)`.  Default: `{}`.
  - `noisynet`: *`None` or `str`*.  Whether or not to enable NoisyNet in building the neural net.  Detailed in the above **Algorithm related options** section.  `str` choices are `['fg', 'ig']` corresponding to factorized and independent Gaussian noises, respectively.  Default: `None`.
  - `save_dir`: *`str`*.  Path to save intermediate `tf.keras` models during training.  Will not save any model if set to `None`.  Defaul: `None`.
  - `save_interval`: *`int`*.  Interval between saving `tf.keras` models during training.  Default: `10000`.
  - `catch_signal`: *`bool`*.  Whether or not to catch `sigint` and `sigterm` during multiprocess training.  Useful in cleaning up dangling processes when run in background but may prevent other parts of the program to respond to signals.  Default: `False`.



## Demo
A minimal demo could be as simple as the following code snippet (in `examples/cartpole_a3c.py`).  (A3C algorithm, `CartPole-v0` environment, and a 2-layer fully-connected net with 200/100 hidden units in each layer.)
```python
'''
cartpole_a3c.py
'''
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
```

## Gym Environment
### Implementing an OpenAI-gym environment maker
The user is supposed to implement a `env_maker` callable which returns **an OpenAI-gym environment**.  Things like history stacking/frame skipping/reward engineering are usually handled here as well.

The above code snippet contains a trivial example:
```python
env_maker=lambda: gym.make('CartPole-v0')
```
which is a callable that returns the `'CartPole-v0'` environment.


## Neural network
### Implementing (part of) a `tf.keras` model
The user is supposed to implement a `feature_maker` callable which takes in an `observation_space` ([explanation](https://gym.openai.com/docs)) and returns `inp_state`, a `tf.keras.layers.Input` layer, and `feature`, a `tf.keras` layer or a tuple of 2 `tf.keras` layers.  For example, with actor-critic algorithms, when `feature` is a `tf.keras` layer, the actor and the critic streams share a common stack of layers. When `feature` is a tuple of 2 `tf.keras` layers, the actor and the critic will be completely separated).

#### Example
The above code snippet `cartpole_a3c.py` also contains a trivial example for the part of a `tf.keras` model:
```python
from tensorflow.python.keras.layers import Input, Dense, Activation

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
```
which makes a fully-connected neural network until the last layer before the policy/value layer.  To use the default feature maker, simply let the feature-maker callable be `feature_maker=lambda obs_space: make_feature(obs_space, [200, 100])`.

### Implementing a full `tf.keras` model
Alternatively, it is possible to specify a full `tf.keras` model by implementing a `model_maker` callable.  `model_maker` should take in the full gym `env` and returns a `tf.keras` model that satisfies the output requirements for each kind of training algorithm.  Its `model.inputs` should always be a 1-tuple like `(inp_state,)` where `inp_state` is a `tf.keras.layers.Input` layer.  Its `model.outputs` should also be a tuple but the content varies according to the selected algorithm.  For example, with `algorithm='a3c'`, `model.outputs` should be a 2-tuple of `(logits, value)`; with `algorithm='dqn'`, `model.outputs` should be a 1-tuple of `(q_values,)`.

#### Example
The following code snippet contains a trivial example for implementing a full `tf.keras` model for A3C or IMPALA:
```python
from tensorflow.python.keras.layers import Input, Dense, Activation
from tensorflow.python.keras.models import Model

'''
Input arguments:
    env:          Gym env;
    num_hid_list: List of hidden unit numbers in the fully-connected net.
'''
def make_feature(env, num_hid_list):
    inp_state = Input(shape=env.observation_space.shape)
    feature = inp_state
    for num_hid in num_hid_list:
        feature = Dense(num_hid)(feature)
        feature = Activation('relu')(feature)
    logits_init = tf.keras.initializers.RandomNormal(stddev=1e-3)
    logits = Dense(env.action_space.n, kernel_initializer=logits_init)(feature)
    value = Dense(1)(feature)
    return Model(inputs=inp_state, outputs=[logits, value])
```
A more detailed usage example can be found in `examples/breakout_acer.py`.

### Implementing an interface function
The user is also supposed to implement a `state_to_input` callable which takes in the `observation` from the output of the OpenAI-gym environment's `reset` or `step` function ([explanation](https://gym.openai.com/docs)) and returns something that a `tf.keras` model can directly take in.  Usually, this function does stuffs like `numpy` stackings/reshapings/etc.  By default, `state_to_input` is set to `None`, in which case the a dummy callable `state_to_input = lambda x: x` will be created and used internally.

**Note:**  So long as `feature_maker` or `model_maker` is implemented correctly, the trainer will run.  However, to utilize the saving/loading functionalities provided by Keras in a hassle-free manner, when writing `feature_maker` or `model_maker` it is recommended to only use combinations of Keras layers that already exist, plus some viable NumPy utilities such as `np.newaxis` (NumPy has to be imported as `import numpy as np` as this is the default importing method assumed by Keras in 'keras/layers/core.py').  It is discouraged to use other modules including plain TensorFlow, as the Keras model loading utility will literally "remember" your code of generating the Keras model and run through the code when it tries to load a saved model.  If we really have to, try to import the needed functionalities **inside** `feature_maker` or `model_maker` so that it will be imported before execution.  However, please do not import the entire TensorFlow (`from tensorflow import x` is fine but no `import tensorflow as tf`) in `feature_maker` or `model_maker` as it will cause circular importing.


