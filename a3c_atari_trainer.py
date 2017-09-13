"""
Atari game deep RL with asynchronous advantage actor-critic (A3C)
Built with distributed tensorflow
"""

''' main function selects running mode '''
def main():
    args = arguments()
    if args.a3c_running_mode == 'trainer':
        trainer(args)
    elif args.a3c_running_mode == 'worker':
        worker(args)


''' arguments block '''
import argparse

def arguments():
    parser = argparse.ArgumentParser(description='A3C Atari')

    # A3C arguments
    parser.add_argument('--a3c_running_mode', default='trainer', type=str,
        choices=['trainer', 'worker'],
        help=argparse.SUPPRESS)
    parser.add_argument('--dtf_num_workers', default=1, type=int,
        help='Total number of parallel workers')
    parser.add_argument('--dtf_worker_index', default=0, type=int,
        help='Index of the current worker')
    parser.add_argument('--dtf_port_begin', default=2220, type=int,
        help='Beginning port used by distributed tensorflow;' + \
             ' ports in range' + \
             ' (dtf_port_begin, dtf_port_begin + dtf_num_workers)' + \
             ' will be occupied')

    # reinforcement learning arguments
    parser.add_argument('--rl_save_path', default='output',
        help='Directory to save data to')
    parser.add_argument('--rl_discount', default=0.99, type=float,
        help='Discount factor gamma')
    parser.add_argument('--rl_learning_rate', default=1e-4, type=float,
        help='Learning rate')
    parser.add_argument('--rl_train_steps', default=1000000, type=int,
        help='Number of training sample interactions with the environment')
    parser.add_argument('--rl_entropy_weight', default=0.01, type=float,
        help='Weight of the entropy term in A3C loss')

    # intervals
    parser.add_argument('--interval_save', default=10000, type=int,
        help='Interval to save weights')

    # rollout arguments
    parser.add_argument('--rollout_maxlen', default=5, type=int,
        help='Maximum length of partial rollout to calculate value target')

    # gym environment arguments
    parser.add_argument('--env', default='Breakout-v0',
        help='Environment name')
    parser.add_argument('--env_resize', nargs=2, type=int, default=(84, 110),
        help='Input shape')
    parser.add_argument('--env_num_frames', default=4, type=int,
        help='Number of frames in a state')
    parser.add_argument('--env_act_steps', default=4, type=int,
        help='Do an action for how many steps')

    # neural net arguments
    parser.add_argument('--net_name', default='fully connected', type=str,
        help='Neural net name')
    parser.add_argument('--net_size', default=512, type=int,
        help='Number of hidden units in the first non-convolutional layer')

    # parse arguments
    args = parser.parse_args()
    args.env_resize = tuple(args.env_resize)
    return args


''' trainer block '''
import time
import subprocess
import signal

class TrainingIndicator(object):
    train = True
    def __init__(self):
        signal.signal(signal.SIGINT, self.handler)
        signal.signal(signal.SIGTERM, self.handler)
    def handler(self, signum, frame):
        self.train = False

def trainer(args):
    num_workers = int(args.dtf_num_workers)
    args_dict = vars(args)
    args_dict['a3c_running_mode'] = 'worker'
    worker_list = []
    for worker_index in range(num_workers):
        args_dict['dtf_worker_index'] = worker_index
        run_list = ['python', __file__]
        for key, value in args_dict.items():
            run_list.append('--{}'.format(key))
            if type(value) == tuple:
                for val in map(str, value):
                    run_list.append(val)
            else:
                run_list.append(str(value))
        worker = subprocess.Popen(run_list, stderr=subprocess.STDOUT)
        worker_list.append(worker)
    train = TrainingIndicator()
    while train.train:
        time.sleep(1)
    for worker in worker_list:
        worker.terminate()
    print('A3C training ends')


''' worker block '''
import os
import signal
import gym
import tensorflow as tf
from a3c.a3c import A3C
from a3c.rollout import Rollout
from a3c.step_counter import StepCounter
from common.envwrapper import Preprocessor, HistoryStacker, RewardClipper
from common.policy import Stochastic
from common.interface import list_frames_to_array
from common.neuralnet.acnet import ACNet
from common.util import get_output_folder
from atari_nets import atari_acnet

def worker(args):
    # environment
    env = gym.make(args.env)
    env = Preprocessor(env, resize=args.env_resize)
    env = HistoryStacker(env, args.env_num_frames, args.env_act_steps)
    env = RewardClipper(env, -1.0, 1.0)
    num_actions = env.action_space.n

    # ports, cluster, and server
    port_list = [args.dtf_port_begin + i for i in range(args.dtf_num_workers)]
    worker_index = args.dtf_worker_index
    port = port_list[worker_index]
    cluster_list = ['localhost:{}'.format(port) for port in port_list]
    cluster = tf.train.ClusterSpec({'local': cluster_list})
    server = tf.train.Server(cluster, job_name='local', task_index=worker_index)
    print('Starting server #{}'.format(worker_index))

    # global/local actor-critic nets
    worker_dev = '/job:local/task:{}/cpu:0'.format(worker_index)
    rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                             cluster=cluster)
    width, height = args.env_resize
    input_shape = height, width, args.env_num_frames
    net_args = input_shape, num_actions, args.net_name, args.net_size

    # global net
    with tf.device(rep_dev):
        acnet_global = ACNet(atari_acnet(*net_args))
        global_weights = acnet_global.weights
        step_counter_global = StepCounter()

    # local net
    with tf.device(worker_dev):
        acnet_local = ACNet(atari_acnet(*net_args))
        acnet_local.set_loss(entropy_weight=args.rl_entropy_weight)
        adam = tf.train.AdamOptimizer(args.rl_learning_rate)
        acnet_local.set_optimizer(adam, train_weights=global_weights)
        acnet_local.set_sync_weights(global_weights)
        step_counter_global.set_increment()

    # policy and rollout
    policy = Stochastic()
    rollout = Rollout(args.rollout_maxlen, num_actions)

    # begin tensorflow session, build a3c agent and train
    with tf.Session('grpc://localhost:{}'.format(port)) as sess:
        is_master = worker_index == 0
        sess.run(tf.global_variables_initializer())
        for obj in acnet_global, acnet_local, step_counter_global:
            obj.set_session(sess)
        agent = A3C(is_master=is_master,
                    acnet_global=acnet_global, acnet_local=acnet_local,
                    state_to_input=list_frames_to_array,
                    policy=policy, rollout=rollout,
                    discount=args.rl_discount,
                    train_steps=args.rl_train_steps,
                    step_counter=step_counter_global,
                    interval_save=args.interval_save)

        # set output path if this is the master worker
        if is_master:
            output = get_output_folder(args.rl_save_path, args.env)
            agent.set_output(output)

        # train the agent
        agent.train(env)

        # terminates the entire training when the master worker terminates
        if is_master:
            print('Master worker terminates')
            os.kill(os.getppid(), signal.SIGTERM)


if __name__ == '__main__':
    main()
