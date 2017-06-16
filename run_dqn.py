#!/usr/bin/env python
"""Play game with DQN."""

import os
import gym
import argparse
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from dqn.dqn import DQN
from dqn.objectives import mean_huber_loss, null_loss
from dqn.policy import *
from dqn.memory import PriorityMemory
from util import get_output_folder
from network.qnetwork import *
from wrapper import *


def main():
    parser = argparse.ArgumentParser(description='Play game with DQN')
    parser.add_argument('--env', default='Pong-v0',
        help='Environment name')
    parser.add_argument('--output', default='output',
        help='Directory to save data to')
    parser.add_argument('--resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--act_steps', default=4, type=int,
        help='Do an action for how many steps')
    parser.add_argument('--mode', default='train', type=str,
        help='Running mode; train/test/rand/video')
    DQN.add_arguments(parser)
    PriorityMemory.add_arguments(parser)
    Policy.add_arguments(parser)
    qnetwork_add_arguments(parser)

    # learning rate for the optimizer
    parser.add_argument('--learning_rate', default=1e-4, type=float,
        help='Learning rate')

    # checkpoint
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')
    parser.add_argument('--read_memory', default=None, type=str,
        help='Read memory from file')

    # parse arguments
    args = parser.parse_args()

    # add new environments here
    if args.env in ['FlappyBird-v0']:
        import gym_ple

    print('########## All arguments:', args)
    args.resize = tuple(args.resize)

    # environment
    env = gym.make(args.env)
    env = PreprocessWrapper(env, resize=args.resize)
    env = HistoryWrapper(env, args.num_frames, args.act_steps)
    num_actions = env.action_space.n

    # online/target q networks, and their interface to env
    width, height = args.resize
    input_shape = height, width, args.num_frames
    online = qnetwork(input_shape, num_actions, args)
    target = qnetwork(input_shape, num_actions, args)
    q_net = {'online': online, 'target': target,
             'interface': interface_list_of_frames}

    # memory
    memory = PriorityMemory(args.dqn_train_steps, args)

    # random, train, test policies
    policy_rand = RandomPolicy(num_actions)
    policy_train = LinearDecayGreedyEpsPolicy(args)
    policy_test = GreedyEpsPolicy(args)
    policy = {'rand': policy_rand, 'train': policy_train, 'test': policy_test}

    # construct and compile the dqn agent
    output = get_output_folder(args.output, args.env)
    agent = DQN(num_actions, q_net, memory, policy, output, args)
    agent.compile([mean_huber_loss, null_loss], Adam(lr=args.learning_rate))

    # read weights/memory if requested
    if args.read_weights is not None:
        online.load_weights(args.read_weights)
    if args.read_memory is not None:
        memory.load(args.read_memory)

    # running
    if args.mode == 'train':
        print('########## training #############')
        agent.train(env)
    elif args.mode == 'test':
        print('########## testing #############')
        agent.test(env)
    elif args.mode == 'rand':
        print('########## random #############')
        agent.random(env)



if __name__ == '__main__':
    main()

