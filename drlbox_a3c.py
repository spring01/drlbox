"""
Asynchronous advantage actor-critic (A3C) trainer
Currently supports only discrete actions
Built with distributed tensorflow
"""

''' main function selects running mode '''
from drlbox.common.manager import Manager

DEFAULT_CONFIG = 'drlbox.config.a3c_default'

def main():
    manager = Manager(description='A3C Trainer', default_config=DEFAULT_CONFIG)

    # a3c specific parser args
    manager.parser.add_argument('--a3c_running_mode', default='trainer',
        type=str, choices=['trainer', 'worker'],
        help='Running mode of this process')
    manager.parser.add_argument('--worker_index', default=0, type=int,
        help='Index of the current worker')

    manager.parse_import()

    if manager.args.a3c_running_mode == 'trainer':
        trainer(manager)
    elif manager.args.a3c_running_mode == 'worker':
        worker(manager)


''' trainer block '''
import subprocess
from drlbox.a3c.blocker import Blocker

def trainer(manager):
    args_dict = vars(manager.args)
    args_dict['a3c_running_mode'] = 'worker'
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
    print('A3C training ends')


''' worker block '''
import os
import signal
import tensorflow as tf
from drlbox.a3c.a3c import A3C
from drlbox.a3c.acnet import ACNet
from drlbox.a3c.rollout import Rollout
from drlbox.a3c.step_counter import StepCounter
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

    # global net
    with tf.device(rep_dev):
        model = manager.build_model(actor_critic_model)
        if is_master:
            model.summary()
        acnet_global = ACNet(model)
        step_counter_global = StepCounter()

    # local net
    with tf.device(worker_dev):
        model = manager.build_model(actor_critic_model)
        acnet_local = ACNet(model)
        acnet_local.set_loss(entropy_weight=config.ENTROPY_WEIGHT)
        adam = tf.train.AdamOptimizer(config.LEARNING_RATE,
                                      epsilon=config.ADAM_EPSILON)
        acnet_local.set_optimizer(adam, train_weights=acnet_global.weights)
        acnet_local.set_sync_weights(acnet_global.weights)
        step_counter_global.set_increment()

    # policy and rollout
    if model.action_mode == 'discrete':
        policy = StochasticDiscrete()
    elif model.action_mode == 'continuous':
        policy = StochasticContinuous(low=manager.env.action_space.low,
                                      high=manager.env.action_space.high)
    else:
        raise ValueError('action_mode not recognized')
    rollout = Rollout(config.ROLLOUT_MAXLEN, config.DISCOUNT)

    # begin tensorflow session, build a3c agent and train
    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        sess.run(tf.global_variables_initializer())
        for obj in acnet_global, acnet_local, step_counter_global:
            obj.set_session(sess)
        agent = A3C(is_master=is_master,
                    acnet_global=acnet_global, acnet_local=acnet_local,
                    state_to_input=manager.state_to_input,
                    policy=policy, rollout=rollout,
                    train_steps=config.TRAIN_STEPS,
                    step_counter=step_counter_global,
                    interval_save=config.INTERVAL_SAVE,
                    output=manager.get_output_folder())
        if args.load_weights is not None:
            acnet_global.load_weights(args.load_weights)

        # train the agent
        agent.train(manager.env)

        # terminates the entire training when the master worker terminates
        if is_master:
            print('Master worker terminates')
            os.kill(os.getppid(), signal.SIGTERM)


if __name__ == '__main__':
    main()
