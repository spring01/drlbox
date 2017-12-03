"""
Deep Q-network (DQN)
Supports deep recurrent Q-network (DRQN) and dueling architecture
"""

import os
import sys
import gym
import argparse
import tensorflow as tf
from hcdrl.dqn.dqn import DQN
from hcdrl.dqn.memory import PriorityMemory
from hcdrl.common.envwrapper import HistoryStacker, RewardClipper
from hcdrl.common.policy import LinearDecayEpsGreedy
from hcdrl.common.neuralnet.qnet import QNet
from hcdrl.common.loss import mean_huber_loss
from hcdrl.common.util import get_output_folder
import importlib


def main():
    parser = argparse.ArgumentParser(description='DQN')

    # gym environment arguments
    parser.add_argument('--env', default='CartPole-v0',
        help='Environment name')
    parser.add_argument('--env_import', default=None,
        help='Some environments need to be registered via import.')
    parser.add_argument('--env_num_frames', default=1, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=1, type=int,
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
    parser.add_argument('--rl_reward_bound', default=[-1.0, 1.0], type=float,
        nargs=2, help='Lower and upper bound for clipped reward')

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

    # train from saved weights/memory
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')
    parser.add_argument('--read_memory', default=None, type=str,
        help='Read memory from file')

    # additional arguments for the nn model
    parser.add_argument('--additional', nargs='+', type=str,
        default=['hcdrl.model.simple_nets', '16 16 16'],
        help='`module additional_args`')

    # path for dynamic imports
    parser.add_argument('--import_path', nargs='+', type=str,
        default=[os.getcwd()],
        help='path where the user-defined scripts are located')

    # parse arguments
    args = parser.parse_args()
    print('########## All arguments:', args)

    # dynamically import net and interface
    for path in args.import_path:
        sys.path.append(path)
    module, additional_args = args.additional[0], args.additional[1:]
    model_spec = importlib.import_module(module)

    # environment
    if args.env_import is not None:
        importlib.import_module(args.env_import)
    env = gym.make(args.env)
    if hasattr(model_spec, 'Preprocessor'):
        env = model_spec.Preprocessor(env)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    lower, upper = args.rl_reward_bound
    env = RewardClipper(env, lower, upper)

    # arguments for building nets
    model_args = env.observation_space, env.action_space, *additional_args

    # online/target q-nets
    model_o, model_t = (model_spec.qnet(*model_args) for _ in range(2))
    model_o.summary()
    online, target = (QNet(model) for model in (model_o, model_t))
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
    agent = DQN(online, target, state_to_input=model_spec.state_to_input,
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

