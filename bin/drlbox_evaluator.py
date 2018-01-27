#!python3
"""
Deep RL evaluator
Support both actor-critic and Q-network
"""

import sys
import gym
import numpy as np
import tensorflow as tf
from drlbox.common.manager import Manager
from drlbox.common.manager import is_discrete_action, is_continuous_action
from drlbox.common.policy import StochasticDiscrete, StochasticContinuous
from drlbox.common.policy import EpsGreedy
from drlbox.net.qnet import QNet
from drlbox.net.acnet import ACNet
from drlbox.net.noisynet import NoisyQNet, NoisyACNet


''' macros '''
DEFAULT_CONFIG = None

def main():
    manager = Manager('Async RL Trainer', default_config=DEFAULT_CONFIG)

    # type of network
    manager.add_argument('--net_type', default='ac',
        choices=['ac', 'dqn'], help='Type of the neural network')
    manager.add_argument('--noisynet', default='false',
        choices=['true', 'false'], help='Invoke NoisyNet when specified')

    # rendering
    manager.add_argument('--render', default='false', type=str,
        choices=['true', 'false', 'end'], help='Rendering option')

    # evaluation
    manager.add_argument('--eval_episodes', default=20, type=int,
        help='Number of episodes in evaluation')
    manager.add_argument('--policy_eps', default=0.01, type=float,
        help='Epsilon in epsilon-greedy policy')

    manager.build_config_env_feature()
    args = manager.args
    action_space = manager.env.action_space

    # network and policy
    if args.net_type == 'ac':
        net_builder = NoisyACNet if args.noisynet == 'true' else ACNet
        if is_discrete_action(action_space):
            policy = StochasticDiscrete()
        elif is_continuous_action(action_space):
            policy = StochasticContinuous(action_space.low, action_space.high)
        else:
            raise ValueError('action_mode not recognized')
    elif args.net_type == 'dqn':
        net_builder = NoisyQNet if args.noisynet == 'true' else QNet
        policy = EpsGreedy(epsilon=args.policy_eps)

    state_feature = manager.build_state_feature()
    net = net_builder(*state_feature, action_space)
    sess = tf.Session()
    net.set_session(sess)
    sess.run(tf.global_variables_initializer())

    if args.load_weights is not None:
        net.load_weights(args.load_weights)

    env = manager.env
    all_total_rewards = []
    for _ in range(args.eval_episodes):
        state = env.reset()
        if args.render == 'true':
            env.render()
        total_rewards = 0.0
        while True:
            state = manager.state_to_input(state)
            action_values = net.action_values(np.stack([state]))[0]
            action = policy.select_action(action_values)
            state, reward, done, info = env.step(action)
            if args.render == 'true':
                env.render()
            total_rewards += reward
            if done:
                break
        if args.render == 'end':
            env.render()
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()
