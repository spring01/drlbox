"""
Deep RL evaluator
Supports both DQN and A3C
"""

import sys
import gym
import argparse
import numpy as np
import tensorflow as tf
from hcdrl.common.envwrapper import HistoryStacker
from hcdrl.common.policy import EpsGreedy, Stochastic
from hcdrl.common.neuralnet.qnet import QNet
from hcdrl.common.neuralnet.acnet import ACNet
import importlib

from hcdrl.model.atari_nets import acnet, qnet


episode_maxlen = 100000

def main():
    parser = argparse.ArgumentParser(description='Deep RL Evaluator')

    # gym environment arguments
    parser.add_argument('--env', default='CartPole-v0',
        help='Environment name')
    parser.add_argument('--env_import', default=None,
        help='Some environments need to be registered via import.')
    parser.add_argument('--env_num_frames', default=1, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=1, type=int,
        help='Do an action for how many steps')

    # policy arguments
    parser.add_argument('--policy_type', default='stochastic', type=str,
        choices=['greedy', 'stochastic'],
        help='Evaluation policy type')
    parser.add_argument('--policy_eps', default=0.01, type=float,
        help='Epsilon in epsilon-greedy policy')

    # neural net arguments
    parser.add_argument('--net_type', default='acnet', type=str,
        choices=['qnet', 'acnet'],
        help='Neural net type')

    # trained weights
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')

    # evaluation
    parser.add_argument('--eval_episodes', default=20, type=int,
        help='Number of episodes in evaluation')

    # rendering
    parser.add_argument('--render', default='true', type=str,
        help='Do rendering or not')

    # additional arguments for the nn model
    parser.add_argument('--additional', nargs='+', type=str,
        default=['hcdrl.model.simple_nets', '16 16 16'],
        help='`module additional_args`')

    # parse arguments
    args = parser.parse_args()
    render = args.render.lower() in ['true', 't']

    print('########## All arguments:', args)

    # dynamically import net and interface
    module, additional_args = args.additional[0], args.additional[1:]
    model_spec = importlib.import_module(module)

    # environment
    if args.env_import is not None:
        importlib.import_module(args.env_import)

    # environment
    env = gym.make(args.env)
    if hasattr(model_spec, 'Preprocessor'):
        env = model_spec.Preprocessor(env)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)

    # arguments for building nets
    model_args = env.observation_space, env.action_space, *additional_args

    # neural net
    net_type = args.net_type.lower()
    if net_type == 'qnet':
        net_builder = lambda x: QNet(model_spec.qnet(*x))
    elif net_type == 'acnet':
        net_builder = lambda x: ACNet(model_spec.acnet(*x))
    net = net_builder(model_args)
    sess = tf.Session()
    net.set_session(sess)
    sess.run(tf.global_variables_initializer())
    net.load_weights(args.read_weights)

    # policy
    if args.policy_type == 'greedy':
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
            state = model_spec.state_to_input(state)
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
