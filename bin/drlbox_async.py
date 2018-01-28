#!python3
"""
Asynchronous trainer built with distributed tensorflow
"""
from drlbox.common.manager import Manager


''' macros '''
DEFAULT_CONFIG = 'drlbox/config/async_default.py' # Config
TRAINER, WORKER = 'trainer', 'worker' # Running mode keywords
KFAC, ADAM = 'kfac', 'adam' # Optimizer type


''' main function selects running mode '''
import argparse
def main():
    manager = Manager('Async RL Trainer', default_config=DEFAULT_CONFIG)

    # async specific parser args
    manager.add_argument('--algorithm', default='a3c',
        choices=['a3c', 'acktr', 'dqn'], help='Training algorithm')
    manager.add_argument('--noisynet', default='false',
        choices=['true', 'false'], help='Invoke NoisyNet when set to true')

    # these arguments are handled internally
    manager.add_argument('--running_mode', default=TRAINER,
        choices=[TRAINER, WORKER], help=argparse.SUPPRESS)
    manager.add_argument('--worker_index', default=0, type=int,
        help=argparse.SUPPRESS)

    manager.build_config_env_feature()

    if manager.args.running_mode == TRAINER:
        call_trainer(manager)
    elif manager.args.running_mode == WORKER:
        call_worker(manager)


''' trainer block '''
import sys
import subprocess
from drlbox.async.blocker import Blocker

def call_trainer(manager):
    manager.args.running_mode = WORKER
    worker_list = []
    for worker_index in range(manager.config.NUM_WORKERS):
        manager.args.worker_index = worker_index
        run_list = [sys.executable, __file__]
        for key, value in vars(manager.args).items():
            if value is not None:
                run_list.append('--{}'.format(key))
                if type(value) is list:
                    for val in map(str, value):
                        run_list.append(val)
                else:
                    run_list.append(str(value))
        worker = subprocess.Popen(run_list, stderr=subprocess.STDOUT)
        worker_list.append(worker)
    Blocker().block()
    for worker in worker_list:
        worker.terminate()
    print('AsyncRL training ends')


''' worker block '''
import os
import signal
import tensorflow as tf
from drlbox.async.async import AsyncRL
from drlbox.common.manager import discrete_action, continuous_action
from drlbox.net import QNet, ACNet, ACKTRNet, NoisyQNet, NoisyACNet
from drlbox.net.kfac import KfacOptimizerTV
from drlbox.async.rollout import RolloutAC, RolloutMultiStepQ
from drlbox.async.step_counter import StepCounter
from drlbox.common.policy import StochasticDisc, StochasticCont, DecayEpsGreedy


def call_worker(manager):
    args, config = manager.args, manager.config
    action_space = manager.env.action_space
    # ports, cluster, and server
    port_list = [config.PORT_BEGIN + i for i in range(config.NUM_WORKERS)]
    worker_index = args.worker_index
    is_master = worker_index == 0
    port = port_list[worker_index]
    cluster_list = ['localhost:{}'.format(port) for port in port_list]
    cluster = tf.train.ClusterSpec({'local': cluster_list})
    server = tf.train.Server(cluster, job_name='local', task_index=worker_index)
    print('Starting server #{}'.format(worker_index))

    # global/local actor-critic nets
    worker_dev = '/job:local/task:{}/cpu:0'.format(worker_index)
    rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                             cluster=cluster)

    # algorithms differ in terms of network structure, rollout, and policy
    if args.algorithm in {'a3c', 'acktr'}:
        loss_kwargs = dict(entropy_weight=config.ENTROPY_WEIGHT,
                           min_var=config.CONT_POLICY_MIN_VAR)
        if args.algorithm == 'a3c':
            net_builder = NoisyACNet if args.noisynet == 'true' else ACNet
            opt_type = ADAM
        elif args.algorithm == 'acktr':
            net_builder = ACKTRNet
            opt_type = KFAC
        build_target = False

        # rollout
        rollout_builder = lambda s: RolloutAC(s, config.DISCOUNT)

        # policy
        if discrete_action(action_space):
            policy = StochasticDisc()
        elif continuous_action(action_space):
            policy = StochasticCont(action_space.low, action_space.high,
                                    min_var=config.CONT_POLICY_MIN_VAR)
    elif args.algorithm == 'dqn':
        loss_kwargs = {}
        net_builder = NoisyQNet if args.noisynet == 'true' else QNet
        opt_type = ADAM
        build_target = True

        # rollout
        rollout_builder = lambda s: RolloutMultiStepQ(s, config.DISCOUNT)

        # policy
        eps_start = config.POLICY_EPS_START
        eps_end = config.POLICY_EPS_END
        eps_delta = (eps_start - eps_end) / config.POLICY_DECAY_STEPS
        policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)

    # global net
    with tf.device(rep_dev):
        if args.load_model is None:
            state, feature = manager.build_state_feature()
            global_net = net_builder.from_sfa(state, feature, action_space)
        else:
            saved_model = net_builder.load_model(args.load_model)
            saved_weights = saved_model.get_weights()
            global_net = net_builder.from_model(saved_model)
        if is_master:
            global_net.model.summary()
        step_counter = StepCounter()


    # local net
    with tf.device(worker_dev):
        state, feature = manager.build_state_feature()
        online_net = net_builder.from_sfa(state, feature, action_space)
        online_net.set_loss(**loss_kwargs)
        if opt_type == KFAC:
            layer_collection = online_net.build_layer_collection()
            opt = KfacOptimizerTV(config.LEARNING_RATE,
                config.KFAC_COV_EMA_DECAY, config.KFAC_DAMPING,
                norm_constraint=config.KFAC_TRUST_RADIUS,
                layer_collection=layer_collection, var_list=online_net.weights)
            online_net.set_optimizer(opt, train_weights=global_net.weights,
                inv_update_interval=config.KFAC_INV_UPD_INTERVAL)
        elif opt_type == ADAM:
            opt = tf.train.AdamOptimizer(config.LEARNING_RATE,
                                         epsilon=config.ADAM_EPSILON)
            online_net.set_optimizer(opt, train_weights=global_net.weights,
                clip_norm=config.GRAD_CLIP_NORM)
        online_net.set_sync_weights(global_net.weights)
        step_counter.set_increment()

    # build a separate global target net for dqn
    if build_target:
        with tf.device(rep_dev):
            state, feature = manager.build_state_feature()
            target_net = net_builder.from_sfa(state, feature, action_space)
            target_net.set_sync_weights(global_net.weights)
    else: # make target net a reference to the local net
        target_net = online_net

    # begin tensorflow session, build async RL agent and train
    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        sess.run(tf.global_variables_initializer())
        for obj in global_net, online_net, step_counter:
            obj.set_session(sess)
        if target_net is not online_net:
            target_net.set_session(sess)
        output = manager.get_output_folder() if is_master else None
        agent = AsyncRL(is_master=is_master,
                        online_net=online_net, target_net=target_net,
                        state_to_input=manager.state_to_input,
                        policy=policy, rollout_builder=rollout_builder,
                        batch_size=config.BATCH_SIZE,
                        train_steps=config.TRAIN_STEPS,
                        step_counter=step_counter,
                        interval_sync_target=config.INTERVAL_SYNC_TARGET,
                        interval_save=config.INTERVAL_SAVE,
                        output=output)
        if args.load_model is not None:
            global_net.set_sync_weights(saved_weights)
            global_net.sync()

        # train the agent
        agent.train(manager.env)

        # terminates the entire training when the master worker terminates
        if is_master:
            print('Master worker terminates')
            os.kill(os.getppid(), signal.SIGTERM)


if __name__ == '__main__':
    main()
