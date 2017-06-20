
import sys
import gym
import argparse
import numpy as np
import tensorflow as tf
from common.envwrapper import Preprocessor, HistoryStacker
from common.policy import EpsGreedy
from common.interface import list_frames_to_array
from common.neuralnet.qnet import QNet, atari_qnet
from common.util import get_output_folder


episode_maxlen = 100000

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
        help='Do an action for how many steps')

    # policy arguments
    parser.add_argument('--policy_eps', default=0.01, type=float,
        help='Epsilon in epsilon-greedy policy')

    # qnet arguments
    parser.add_argument('--qnet_name', default='dqn', type=str,
        help='Q-net name')
    parser.add_argument('--qnet_size', default=512, type=int,
        help='Number of hidden units in the first non-convolutional layer')

    # trained weights
    parser.add_argument('--read_weights', default=None, type=str,
        help='Read weights from file')

    # evaluation
    parser.add_argument('--eval_episodes', default=20, type=int,
        help='Number of episodes in evaluation')

    # parse arguments
    args = parser.parse_args()

    # add new environments here
    if args.env in ['FlappyBird-v0']:
        import gym_ple

    print('########## All arguments:', args)
    args.env_resize = tuple(args.env_resize)

    # environment
    env = gym.make(args.env)
    env = Preprocessor(env, resize=args.env_resize)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    num_actions = env.action_space.n

    # q-net
    width, height = args.env_resize
    input_shape = height, width, args.env_num_frames
    qnet_args = input_shape, num_actions, args.qnet_name, args.qnet_size
    qnet = QNet(atari_qnet(*qnet_args))
    sess = tf.Session()
    qnet.set_session(sess)
    sess.run(tf.global_variables_initializer())
    qnet.load_weights(args.read_weights)

    # policy
    policy = EpsGreedy(epsilon=args.policy_eps)

    all_total_rewards = []
    for _ in range(args.eval_episodes):
        state = env.reset()
        total_rewards = 0.0
        for i in range(episode_maxlen):
            state = list_frames_to_array(state)
            action_values = qnet.action_values(np.stack([state]))[0]
            action = policy.select_action(action_values)
            state, reward, done, info = env.step(action)
            total_rewards += reward
            if done:
                break
        all_total_rewards.append(total_rewards)
        print('episode reward:', total_rewards)
    print('average episode reward:', np.mean(all_total_rewards))


if __name__ == "__main__":
    main()
