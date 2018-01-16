"""
Deep RL evaluator
Support both actor-critic and Q-network
"""

import sys
import gym
import numpy as np
import tensorflow as tf
from drlbox.common.manager import Manager, DISCRETE, CONTINUOUS
from drlbox.common.policy import StochasticDiscrete, StochasticContinuous
from drlbox.common.policy import EpsGreedy
from drlbox.dqn.qnet import QNet
from drlbox.async.acnet import ACNet
from drlbox.model.actor_critic import actor_critic_model
from drlbox.model.q_network import q_network_model

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

    # network and policy
    if args.net_type == 'ac':
        net_builder = ACNet
        model_func = actor_critic_model
        if manager.action_mode == DISCRETE:
            policy = StochasticDiscrete()
        elif manager.action_mode == CONTINUOUS:
            action_space = manager.env.action_space
            policy = StochasticContinuous(action_space.low, action_space.high)
        else:
            raise ValueError('action_mode not recognized')
    elif args.net_type == 'dqn':
        net_builder = QNet
        model_func = q_network_model
        policy = EpsGreedy(epsilon=args.policy_eps)

    # invoke NoisyNet if specified
    if args.noisynet == 'true':
        model_builder = lambda *x: model_func(*x, noisy=True)
    else:
        model_builder = model_func

    model = manager.build_model(model_builder)
    net = net_builder(model)
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
