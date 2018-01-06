"""
Asynchronous trainer built with distributed tensorflow
"""

''' main function selects running mode '''
from drlbox.common.manager import Manager

DEFAULT_CONFIG = 'drlbox.config.async_default'

def main():
    manager = Manager('Async RL Trainer', default_config=DEFAULT_CONFIG)

    # async specific parser args
    manager.parser.add_argument('--algorithm', default='a3c',
        type=str, choices=['a3c', 'acktr', 'dqn'],
        help='Training algorithm')
    manager.parser.add_argument('--async_running_mode', default='trainer',
        type=str, choices=['trainer', 'worker'],
        help='Running mode of this process')
    manager.parser.add_argument('--worker_index', default=0, type=int,
        help='Index of the current worker')

    manager.parse_import()

    if manager.args.async_running_mode == 'trainer':
        trainer(manager)
    elif manager.args.async_running_mode == 'worker':
        worker(manager)


''' trainer block '''
import subprocess
from drlbox.async.blocker import Blocker

def trainer(manager):
    args_dict = vars(manager.args)
    args_dict['async_running_mode'] = 'worker'
    worker_list = []
    for worker_index in range(manager.config.NUM_WORKERS):
        args_dict['worker_index'] = worker_index
        run_list = ['python', __file__]
        for key, value in args_dict.items():
            if value is not None:
                run_list.append('--{}'.format(key))
                if type(value) == list:
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
from drlbox.async.acnet import ACNet
from drlbox.async.acktrnet import ACKTRNet
from drlbox.async.kfac import KfacOptimizerTV
from drlbox.async.rollout import RolloutAC, RolloutQ
from drlbox.async.step_counter import StepCounter
from drlbox.common.policy import StochasticDiscrete, StochasticContinuous
from drlbox.model.actor_critic import actor_critic_model


def worker(manager):
    args, config = manager.args, manager.config
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

    # determine training algorithm
    algorithm = args.algorithm.lower()
    if algorithm == 'a3c':
        net_builder = ACNet
        rollout_builder = RolloutAC
    elif algorithm == 'acktr':
        net_builder = lambda mod: ACKTRNet(mod, config.KFAC_INV_UPD_INTERVAL)
        rollout_builder = RolloutAC
    elif algorithm == 'dqn':
        net_builder = QNet
        rollout_builder = RolloutQ

    # global net
    with tf.device(rep_dev):
        model = manager.build_model(actor_critic_model)
        if is_master:
            model.summary()
        global_net = net_builder(model)
        step_counter = StepCounter()

    # local net
    with tf.device(worker_dev):
        model = manager.build_model(actor_critic_model)
        local_net = net_builder(model)
        local_net.set_loss(entropy_weight=config.ENTROPY_WEIGHT)
        lr = config.LEARNING_RATE

        if algorithm == 'acktr':
            layer_collection = local_net.build_layer_collection(model)
            opt = KfacOptimizerTV(lr, config.KFAC_COV_EMA_DECAY,
                config.KFAC_DAMPING, norm_constraint=config.KFAC_TRUST_RADIUS,
                layer_collection=layer_collection, var_list=local_net.weights)
            local_net.set_optimizer(opt, train_weights=global_net.weights)
        else:
            opt = tf.train.AdamOptimizer(lr, epsilon=config.ADAM_EPSILON)
            local_net.set_optimizer(opt, train_weights=global_net.weights,
                                    clip_norm=config.GRAD_CLIP_NORM)
        local_net.set_sync_weights(global_net.weights)
        step_counter.set_increment()

    # policy and rollout
    if algorithm == 'a3c' or algorithm == 'acktr':
        if model.action_mode == 'discrete':
            policy = StochasticDiscrete()
        elif model.action_mode == 'continuous':
            act_space = manager.env.action_space
            policy = StochasticContinuous(act_space.low, act_space.high)
        else:
            raise ValueError('action_mode not recognized')
    elif algorthm == 'dqn':
        # todo: change decay method to explicit step dependent
        eps_start = config.POLICY_EPS_START
        eps_end = config.POLICY_EPS_END
        eps_delta = (eps_start - eps_end) / config.POLICY_DECAY_STEPS
        policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)
    rollout = rollout_builder(config.ROLLOUT_MAXLEN, config.DISCOUNT)

    # begin tensorflow session, build async RL agent and train
    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        sess.run(tf.global_variables_initializer())
        for obj in global_net, local_net, step_counter:
            obj.set_session(sess)
        agent = AsyncRL(is_master=is_master, local_net=local_net,
                        state_to_input=manager.state_to_input,
                        policy=policy, rollout=rollout,
                        batch_size=config.BATCH_SIZE,
                        train_steps=config.TRAIN_STEPS,
                        step_counter=step_counter,
                        interval_save=config.INTERVAL_SAVE,
                        output=manager.get_output_folder())
        if args.load_weights is not None:
            global_net.load_weights(args.load_weights)

        # train the agent
        agent.train(manager.env)

        # terminates the entire training when the master worker terminates
        if is_master:
            print('Master worker terminates')
            os.kill(os.getppid(), signal.SIGTERM)


if __name__ == '__main__':
    main()