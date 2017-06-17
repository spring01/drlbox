#!/usr/bin/env python

import os
import gym
import argparse
from tensorflow.contrib.keras.api.keras.optimizers import Adam
from dqn.dqn import DQN
from dqn.objectives import mean_huber_loss
from dqn.policy import LinearDecayGreedyEpsPolicy
from dqn.memory import PriorityMemory
from common.util import get_output_folder
from common.neuralnet.qnet import build_qnet
from common.interface import list_frames_to_array
from common.envwrapper import Preprocessor, HistoryStacker, RewardClipper


def main():
    parser = argparse.ArgumentParser(description='DQN')

    # output path
    parser.add_argument('--output', default='output',
        help='Directory to save data to')

    # gym environment arguments
    parser.add_argument('--env', default='Pong-v0',
        help='Environment name')
    parser.add_argument('--resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # dqn arguments
    parser.add_argument('--dqn_discount', default=0.99, type=float,
        help='Discount factor gamma')
    parser.add_argument('--dqn_train_steps', default=2000000, type=int,
        help='Number of training sample interactions with the environment')

    # memory arguments
    parser.add_argument('--memory_maxlen', default=500000, type=int,
        help='Replay memory length')
    parser.add_argument('--memory_fill', default=50000, type=int,
        help='Fill the replay memory to how much length before update')
    parser.add_argument('--memory_alpha', default=0.6, type=float,
        help='Exponent alpha in prioritized replay memory')
    parser.add_argument('--memory_beta0', default=0.4, type=float,
        help='Initial beta in prioritized replay memory')

    # policy arguments
    parser.add_argument('--policy_start_eps', default=1.0, type=float,
        help='Starting probability in linear-decay epsilon-greedy')
    parser.add_argument('--policy_end_eps', default=0.1, type=float,
        help='Ending probability in linear-decay epsilon-greedy')
    parser.add_argument('--policy_decay_steps', default=500000, type=int,
        help='Decay steps in linear-decay epsilon-greedy')

    # qnet arguments
    parser.add_argument('--qnet_name', default='dqn', type=str,
        help='Q-net name')
    parser.add_argument('--qnet_size', default=256, type=int,
        help='Number of hidden units in the first non-convolutional layer')

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
    env = Preprocessor(env, resize=args.resize)
    env = HistoryStacker(env, args.num_frames, args.act_steps)
    env = RewardClipper(env, -1.0, 1.0)
    num_actions = env.action_space.n

    # online/target q-nets
    width, height = args.resize
    input_shape = height, width, args.num_frames
    qnet_args = input_shape, num_actions, args.qnet_name, args.qnet_size
    online, target = (build_qnet(*qnet_args) for _ in range(2))
    online.compile(loss=mean_huber_loss, optimizer=Adam(lr=args.learning_rate))
    target.compile(loss=mean_huber_loss, optimizer=Adam(lr=args.learning_rate))

    # memory and policy
    memory = PriorityMemory(train_steps=args.dqn_train_steps,
                            maxlen=args.memory_maxlen,
                            fill=args.memory_fill,
                            alpha=args.memory_alpha,
                            beta0=args.memory_beta0)
    policy = LinearDecayGreedyEpsPolicy(start_eps=args.policy_start_eps,
                                        end_eps=args.policy_end_eps,
                                        decay_steps=args.policy_decay_steps)

    # construct and compile the dqn agent
    output = get_output_folder(args.output, args.env)
    agent = DQN(num_actions=num_actions, online=online, target=target,
                state_to_input=list_frames_to_array,
                output=output, memory=memory, policy=policy,
                discount=args.dqn_discount, train_steps=args.dqn_train_steps)

    # read weights/memory if requested
    if args.read_weights is not None:
        online.load_weights(args.read_weights)
    if args.read_memory is not None:
        memory.load(args.read_memory)

    # train the agent
    agent.train(env)




if __name__ == '__main__':
    main()

