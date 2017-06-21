"""
Atari game deep RL evaluator
Supports both DQN and A3C
"""

import sys
import gym
import argparse
import numpy as np
import tensorflow as tf
from common.envwrapper import Preprocessor, HistoryStacker
from common.interface import list_frames_to_array
from common.policy import EpsGreedy, Stochastic
from common.neuralnet.qnet import QNet, atari_qnet
from common.neuralnet.acnet import ACNet, atari_acnet
from common.util import get_output_folder


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='Deep RL Atari')

    # gym environment arguments
    parser.add_argument('--env', default='Breakout-v0',
        help='Environment name')
    parser.add_argument('--env_resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--env_num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # policy arguments
    parser.add_argument('--policy_type', default='epsilon greedy', type=str,
        choices=['epsilon greedy', 'stochastic'],
        help='Evaluation policy type')
    parser.add_argument('--policy_eps', default=0.01, type=float,
        help='Epsilon in epsilon-greedy policy')

    # neural net arguments
    parser.add_argument('--net_type', default='qnet', type=str,
        choices=['qnet', 'acnet'],
        help='Neural net type')
    parser.add_argument('--net_name', default='dqn', type=str,
        help='Neural net name')
    parser.add_argument('--net_size', default=512, type=int,
        help='Number of hidden units in the first non-convolutional layer')

    # trained weights
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')

    # evaluation
    parser.add_argument('--eval_episodes', default=20, type=int,
        help='Number of episodes in evaluation')

    # rendering
    parser.add_argument('--render', default='true', type=str,
        choices=['true', 'True', 't', 'T', 'false', 'False', 'f', 'F'],
        help='Do rendering or not')

    # parse arguments
    args = parser.parse_args()
    render = args.render.lower() in ['true', 't']

    print('########## All arguments:', args)
    args.env_resize = tuple(args.env_resize)

    # environment
    env = gym.make(args.env)
    env = Preprocessor(env, resize=args.env_resize)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    num_actions = env.action_space.n

    # neural net
    net_type = args.net_type.lower()
    if net_type == 'qnet':
        net_builder = lambda args: QNet(atari_qnet(*args))
    elif net_type == 'acnet':
        net_builder = lambda args: ACNet(atari_acnet(*args))
    width, height = args.env_resize
    input_shape = height, width, args.env_num_frames
    net_args = input_shape, num_actions, args.net_name, args.net_size
    net = net_builder(net_args)
    sess = tf.Session()
    net.set_session(sess)
    sess.run(tf.global_variables_initializer())
    net.load_weights(args.read_weights)

    # policy
    if args.policy_type == 'epsilon greedy':
        policy = EpsGreedy(epsilon=args.policy_eps)
    elif args.policy_type == 'stochastic':
        policy = Stochastic()

    all_total_rewards = []
    for _ in range(args.eval_episodes):
        state = env.reset()
        if render:
            env.render()
        total_rewards = 0.0
        for i in range(episode_maxlen):
            state = list_frames_to_array(state)
            action_values = net.action_values(np.stack([state]))[0]
            action = policy.select_action(action_values)
            state, reward, done, info = env.step(action)
            if render:
                env.render()
            total_rewards += reward
            if done:
                break
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()
