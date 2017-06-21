"""
Atari game deep RL with deep Q-network (DQN)
Supports deep recurrent Q-network (DRQN) and dueling architecture
"""

import os
import gym
import argparse
import tensorflow as tf
from dqn.dqn import DQN
from dqn.memory import PriorityMemory
from common.envwrapper import Preprocessor, HistoryStacker, RewardClipper
from common.policy import LinearDecayEpsGreedy
from common.interface import list_frames_to_array
from common.neuralnet.qnet import QNet, atari_qnet
from common.loss import mean_huber_loss
from common.util import get_output_folder


def main():
    parser = argparse.ArgumentParser(description='DQN Atari')

    # gym environment arguments
    parser.add_argument('--env', default='Breakout-v0',
        help='Environment name')
    parser.add_argument('--env_resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--env_num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps before observing')

    # reinforcement learning arguments
    parser.add_argument('--rl_save_path', default='./output',
        help='Directory to save data to')
    parser.add_argument('--rl_discount', default=0.99, type=float,
        help='Discount factor gamma')
    parser.add_argument('--rl_learning_rate', default=1e-4, type=float,
        help='Learning rate')
    parser.add_argument('--rl_train_steps', default=1000000, type=int,
        help='Number of training sample interactions with the environment')

    # intervals
    parser.add_argument('--interval_train_online', default=4, type=int,
        help='Interval to train the online network')
    parser.add_argument('--interval_sync_target', default=40000, type=int,
        help='Interval to reset the target network')
    parser.add_argument('--interval_save', default=40000, type=int,
        help='Interval to save weights and memory')

    # memory arguments
    parser.add_argument('--memory_maxlen', default=100000, type=int,
        help='Replay memory length')
    parser.add_argument('--memory_fill', default=10000, type=int,
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

    # neural net arguments
    parser.add_argument('--net_name', default='dqn', type=str,
        help='Neural net name')
    parser.add_argument('--net_size', default=512, type=int,
        help='Number of hidden units in the first non-convolutional layer')

    # checkpoint
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')
    parser.add_argument('--read_memory', default=None, type=str,
        help='Read memory from file')

    # parse arguments
    args = parser.parse_args()

    print('########## All arguments:', args)
    args.env_resize = tuple(args.env_resize)

    # environment
    env = gym.make(args.env)
    env = Preprocessor(env, resize=args.env_resize)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    env = RewardClipper(env, -1.0, 1.0)
    num_actions = env.action_space.n

    # online/target q-nets
    width, height = args.env_resize
    input_shape = height, width, args.env_num_frames
    qnet_args = input_shape, num_actions, args.net_name, args.net_size
    online, target = (QNet(atari_qnet(*qnet_args)) for _ in range(2))
    sess = tf.Session()
    for net in online, target:
        net.set_loss(mean_huber_loss)
        net.set_optimizer(tf.train.AdamOptimizer(args.rl_learning_rate))
        net.set_session(sess)
    target.set_sync_weights(online.weights)
    sess.run(tf.global_variables_initializer())

    # memory and policy
    memory = PriorityMemory(train_steps=args.rl_train_steps,
                            maxlen=args.memory_maxlen,
                            fill=args.memory_fill,
                            alpha=args.memory_alpha,
                            beta0=args.memory_beta0)
    policy = LinearDecayEpsGreedy(start_eps=args.policy_start_eps,
                                  end_eps=args.policy_end_eps,
                                  decay_steps=args.policy_decay_steps)

    # construct and compile the dqn agent
    output = get_output_folder(args.rl_save_path, args.env)
    agent = DQN(online, target, state_to_input=list_frames_to_array,
                memory=memory, policy=policy,
                discount=args.rl_discount, train_steps=args.rl_train_steps,
                interval_train_online=args.interval_train_online,
                interval_sync_target=args.interval_sync_target,
                interval_save=args.interval_save,
                output=output)

    # read weights/memory if requested
    if args.read_weights is not None:
        online.load_weights(args.read_weights)
    if args.read_memory is not None:
        memory.load(args.read_memory)

    # train the agent
    agent.train(env)


if __name__ == '__main__':
    main()

