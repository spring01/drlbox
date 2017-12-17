"""
Asynchronous advantage actor-critic (A3C) trainer
Currently supports only discrete actions
Built with distributed tensorflow
"""

''' main function selects running mode '''
import sys
import importlib

def main():
    args = arguments()
    # dynamically import net and interface
    for path in args.import_path:
        sys.path.append(path)
    config = importlib.import_module(args.import_config)
    if args.a3c_running_mode == 'trainer':
        trainer(args, config)
    elif args.a3c_running_mode == 'worker':
        worker(args, config)


''' arguments block '''
import argparse
import os

def arguments():
    parser = argparse.ArgumentParser(description='A3C Trainer')

    parser.add_argument('--a3c_running_mode', default='trainer', type=str,
        choices=['trainer', 'worker'],
        help=argparse.SUPPRESS)
    parser.add_argument('--worker_index', default=0, type=int,
        help='Index of the current worker')
    parser.add_argument('--load_weights', default=None,
        help='If specified, load weights and start training from there')

    # user-definable imports
    parser.add_argument('--import_path', nargs='+', default=[os.getcwd()],
        help='path where the user-defined scripts are located')
    parser.add_argument('--import_env', nargs='+', default=None,
        help='openai gym environment.')
    parser.add_argument('--import_model', nargs='+',
        default=['drlbox.model.fc_ac', '200 100'],
        help='neural network model')
    parser.add_argument('--import_config', default='drlbox.config.a3c_debug',
        help='algorithm configurations')

    # parse arguments
    args = parser.parse_args()
    print('########## All arguments:', args)
    return args




''' trainer block '''
import subprocess
from drlbox.a3c.blocker import Blocker

def trainer(args, config):
    args_dict = vars(args)
    args_dict['a3c_running_mode'] = 'worker'
    worker_list = []
    for worker_index in range(config.NUM_WORKERS):
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
import signal
import tensorflow as tf
from drlbox.a3c.a3c import A3C
from drlbox.a3c.acnet import ACNet
from drlbox.a3c.rollout import Rollout
from drlbox.a3c.step_counter import StepCounter
from drlbox.common.policy import StochasticDiscrete
from drlbox.common.util import get_output_folder


def worker(args, config):
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

    # gym environment
    if args.import_env is None:
        import_env = 'drlbox.env.default'
        env_args = []
    else:
        import_env = args.import_env[0]
        env_args = args.import_env[1:]
    env_spec = importlib.import_module(import_env)
    env, env_name = env_spec.make_env(*env_args)

    # tensorflow-keras model
    model_spec = importlib.import_module(args.import_model[0])
    model_args = env.observation_space, env.action_space, *args.import_model[1:]

    # global net
    with tf.device(rep_dev):
        model = model_spec.model(*model_args)
        if is_master:
            model.summary()
        acnet_global = ACNet(model)
        global_weights = acnet_global.weights
        step_counter_global = StepCounter()

    # local net
    with tf.device(worker_dev):
        acnet_local = ACNet(model_spec.model(*model_args))
        acnet_local.set_loss(entropy_weight=config.RL_ENTROPY_WEIGHT)
        adam = tf.train.AdamOptimizer(config.RL_LEARNING_RATE)
        acnet_local.set_optimizer(adam, train_weights=global_weights)
        acnet_local.set_sync_weights(global_weights)
        step_counter_global.set_increment()

    # policy and rollout
    policy = StochasticDiscrete()
    rollout = Rollout(config.RL_ROLLOUT_MAXLEN, config.RL_DISCOUNT)

    # begin tensorflow session, build a3c agent and train
    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        sess.run(tf.global_variables_initializer())
        for obj in acnet_global, acnet_local, step_counter_global:
            obj.set_session(sess)
        agent = A3C(is_master=is_master,
                    acnet_global=acnet_global, acnet_local=acnet_local,
                    state_to_input=model_spec.state_to_input,
                    policy=policy, rollout=rollout,
                    train_steps=config.RL_TRAIN_STEPS,
                    step_counter=step_counter_global,
                    interval_save=config.RL_INTERVAL_SAVE)

        # set output path if this is the master worker
        if is_master:
            output = get_output_folder(config.SAVE_PATH, env_name)
            agent.set_output(output)
        if args.load_weights is not None:
            acnet_global.load_weights(args.load_weights)

        # train the agent
        agent.train(env)

        # terminates the entire training when the master worker terminates
        if is_master:
            print('Master worker terminates')
            os.kill(os.getppid(), signal.SIGTERM)


if __name__ == '__main__':
    main()