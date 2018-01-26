#!python3
"""
Deep Q-network (DQN)
Supports deep recurrent Q-network (DRQN) and dueling architecture
"""

import tensorflow as tf
from drlbox.dqn.dqn import DQN
from drlbox.dqn.qnet import QNet
from drlbox.dqn.replay import Replay, PriorityReplay
from drlbox.common.manager import Manager
from drlbox.common.manager import is_discrete_action, is_continuous_action
from drlbox.common.policy import DecayEpsGreedy


DEFAULT_CONFIG = 'drlbox/config/dqn_default.py'

def main():
    manager = Manager(description='DQN Trainer', default_config=DEFAULT_CONFIG)

    # dqn specific parser args
    manager.add_argument('--load_replay', default=None,
        help='If specified, load replay memory and start training from there')

    manager.build_config_env_feature()
    args, config = manager.args, manager.config

    # online/target q-nets
    online_state_feature = manager.build_state_feature()
    online = QNet(*online_state_feature, manager.env.action_space)
    target_state_feature = manager.build_state_feature()
    target = QNet(*target_state_feature, manager.env.action_space)
    online.model.summary()
    sess = tf.Session()
    for net in online, target:
        net.set_loss()
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
    agent = DQN(online, target, state_to_input=manager.state_to_input,
                replay=replay, policy=policy, discount=config.DISCOUNT,
                train_steps=config.TRAIN_STEPS, batch_size=config.BATCH_SIZE,
                interval_train_online=config.INTERVAL_TRAIN_ONLINE,
                interval_sync_target=config.INTERVAL_SYNC_TARGET,
                interval_save=config.INTERVAL_SAVE,
                output=manager.get_output_folder())

    # train the agent
    agent.train(manager.env)


if __name__ == '__main__':
    main()

