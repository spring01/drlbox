"""
Deep Q-network (DQN)
Supports deep recurrent Q-network (DRQN) and dueling architecture
"""

import os
import sys
import gym
import argparse
import tensorflow as tf
from drlbox.dqn.dqn import DQN
from drlbox.dqn.qnet import QNet
from drlbox.dqn.replay import Replay, PriorityReplay
from drlbox.common.policy import DecayEpsGreedy
from drlbox.common.loss import mean_huber_loss
from drlbox.common.util import get_output_folder
from drlbox.model.q_network import q_network_model
import importlib


DEFAULT_CONFIG = 'drlbox.config.dqn_default'

def main():
    parser = argparse.ArgumentParser(description='DQN')
    parser.add_argument('--save', default='./output',
        help='Directory to save data to')

    # train from saved weights/replay
    parser.add_argument('--load_weights', default=None, type=str,
        help='If specified, load weights and start training from there')
    parser.add_argument('--load_replay', default=None, type=str,
        help='If specified, load replay memory and start training from there')

    # user-definable imports
    parser.add_argument('--import_path', nargs='+', default=[os.getcwd()],
        help='path where the user-defined scripts are located')
    parser.add_argument('--import_env', nargs='+',
        default=['drlbox.env.default', 'CartPole-v0'],
        help='openai gym environment.')
    parser.add_argument('--import_feature', nargs='+',
        default=['drlbox.feature.fc', '200 100'],
        help='neural network feature builder')
    parser.add_argument('--import_config', default=DEFAULT_CONFIG,
        help='algorithm configurations')

    # parse arguments
    args = parser.parse_args()
    print('########## All arguments:', args)

    # dynamically import net and interface
    for path in args.import_path:
        sys.path.append(path)
    config_def = importlib.import_module(DEFAULT_CONFIG)
    config = importlib.import_module(args.import_config)

    # set default configurations in config
    for key, value in config_def.__dict__.items():
        if key not in config.__dict__:
            config.__dict__[key] = value

    # gym environment
    env_spec = importlib.import_module(args.import_env[0])
    env, env_name = env_spec.make_env(*args.import_env[1:])
    action_space = env.action_space

    # tensorflow-keras feature builder
    feature_spec = importlib.import_module(args.import_feature[0])
    feature_args = env.observation_space, *args.import_feature[1:]

    # online/target q-nets
    state_o, feature_o = feature_spec.feature(*feature_args)
    model_o = q_network_model(state_o, feature_o, action_space)
    state_t, feature_t = feature_spec.feature(*feature_args)
    model_t = q_network_model(state_t, feature_t, action_space)

    model_o.summary()
    online, target = (QNet(model) for model in (model_o, model_t))
    sess = tf.Session()
    for net in online, target:
        net.set_loss(mean_huber_loss)
        adam = tf.train.AdamOptimizer(config.LEARNING_RATE,
                                      epsilon=config.ADAM_EPSILON)
        net.set_optimizer(adam)
        net.set_session(sess)
    target.set_sync_weights(online.weights)
    sess.run(tf.global_variables_initializer())

    # replay memory
    maxlen, minlen = config.REPLAY_MAXLEN, config.REPLAY_MINLEN
    if config.REPLAY_TYPE == 'priority':
        beta_delta = (1.0 - config.REPLAY_BETA) / config.TRAIN_STEPS
        replay = PriorityReplay(maxlen, minlen,
                                alpha=config.REPLAY_ALPHA,
                                beta=config.REPLAY_BETA,
                                beta_delta=beta_delta)
    else:
        replay = Replay(maxlen, minlen)

    # policy
    eps_start = config.POLICY_EPS_START
    eps_end = config.POLICY_EPS_END
    eps_delta = (eps_start - eps_end) / config.POLICY_DECAY_STEPS
    policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)

    # read weights/memory if requested
    if args.load_weights is not None:
        online.load_weights(args.load_weights)
    if args.load_replay is not None:
        replay = Replay.load(args.load_replay)

    # construct and compile the dqn agent
    output = get_output_folder(args.save, env_name)
    agent = DQN(online, target, state_to_input=feature_spec.state_to_input,
                replay=replay, policy=policy, discount=config.DISCOUNT,
                train_steps=config.TRAIN_STEPS, batch_size=config.BATCH_SIZE,
                interval_train_online=config.INTERVAL_TRAIN_ONLINE,
                interval_sync_target=config.INTERVAL_SYNC_TARGET,
                interval_save=config.INTERVAL_SAVE,
                output=output)

    # train the agent
    agent.train(env)


if __name__ == '__main__':
    main()

