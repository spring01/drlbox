
from multiprocessing import Process, cpu_count
import socket

import os
import time
from datetime import timedelta

import tensorflow as tf
import builtins
from numpy import concatenate, zeros
from drlbox.common.util import set_args
from .step_counter import StepCounter
from .rollout import Rollout


print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

LOCALHOST = 'localhost'
JOBNAME = 'local'

class Trainer:

    KEYWORD_DICT = dict(env_maker=None,
                        feature_maker=None,
                        state_to_input=None,
                        load_model=None,
                        save_dir=None,              # Directory to save data to
                        num_parallel=cpu_count(),
                        port_begin=2220,
                        discount=0.99,
                        train_steps=1000000,
                        opt_learning_rate=1e-4,
                        opt_batch_size=32,
                        opt_adam_epsilon=1e-4,
                        opt_grad_clip_norm=40.0,
                        interval_save=10000,)

    def __init__(self, **kwargs):
        set_args(self, self.KEYWORD_DICT, kwargs)

    def run(self):
        self.port_list = [self.port_begin + i for i in range(self.num_parallel)]
        for port in self.port_list:
            if not self.port_available(LOCALHOST, port):
                raise NameError('port {} is not available'.format(port))
        print('Claiming {} port {} ...'.format(LOCALHOST, self.port_list))
        worker_list = []
        for wid in range(self.num_parallel):
            worker = Process(target=self.worker, args=(wid,))
            worker.start()
            worker_list.append(worker)

        # terminates the entire training when the master worker terminates
        master_worker = worker_list[0]
        wait_counter = 0
        start_time = time.time()
        while master_worker.is_alive():
            wait_counter += 1
            if wait_counter >= 3000:
                wait_counter = 0
                elapsed = int(time.time() - start_time)
                time_str = str(timedelta(seconds=elapsed))
                print('Elapsed time:', time_str)
            time.sleep(0.1)
        print('Master worker terminated -- training should end soon')
        for worker in worker_list:
            worker.terminate()
        print('Asynchronous training has ended')

    def worker(self, wid):
        env = self.env_maker()
        self.output = self.get_output_dir(env.spec.id)

        # ports, cluster, and server
        self.is_master = wid == 0
        cluster_list = ['{}:{}'.format(LOCALHOST, p) for p in self.port_list]
        cluster = tf.train.ClusterSpec({JOBNAME: cluster_list})
        server = tf.train.Server(cluster, job_name=JOBNAME, task_index=wid)
        print('Starting server #{}'.format(wid))

        self.setup_algorithm(env.action_space)

        # global/local devices
        worker_dev = '/job:{}/task:{}/cpu:0'.format(JOBNAME, wid)
        rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                                 cluster=cluster)

        self.setup_nets(worker_dev, rep_dev, env)

        # begin tensorflow session, build async RL agent and train
        port = self.port_list[wid]
        with tf.Session('grpc://{}:{}'.format(LOCALHOST, port)) as sess:
            sess.run(tf.global_variables_initializer())
            self.set_session(sess)

            # train the agent
            self.train_on_env(env)

    def train_on_env(self, env):
        step = self.step_counter.step_count()
        if self.is_master:
            last_save = step
            self.save_model(step)

        state = env.reset()
        state = self.state_to_input(state)
        episode_reward = 0.0
        while step <= self.train_steps:
            self.online_net.sync()
            rollout_list = [Rollout(state)]
            for batch_step in range(self.opt_batch_size):
                act_val = self.online_net.action_values([state])[0]
                action = self.policy.select_action(act_val)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                state = self.state_to_input(state)
                rollout_list[-1].append(state, action, reward, done, act_val)
                if done:
                    state = env.reset()
                    state = self.state_to_input(state)
                    if batch_step < self.opt_batch_size - 1:
                        rollout_list.append(Rollout(state))
                    print('episode reward {:5.2f}'.format(episode_reward))
                    episode_reward = 0.0

            batch_loss = self.train_on_rollout_list(rollout_list)

            self.step_counter.increment(self.opt_batch_size)
            step = self.step_counter.step_count()
            if self.is_master:
                if step - last_save > self.interval_save:
                    self.save_model(step)
                    last_save = step
                str_step = 'training step {}/{}'.format(step, self.train_steps)
                print(str_step + ', loss {:3.3f}'.format(batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_model(step)

    def port_available(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex((host, port)) != 0

    def build_net(self, env):
        state, feature = self.feature_maker(env.observation_space)
        return self.net_cls.from_sfa(state, feature, env.action_space)

    def get_output_dir(self, env_name):
        if self.save_dir is None:
            return None
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            print('Made output dir', self.save_dir)
        save_dir = self.save_dir
        experiment_id = 0
        for folder_name in os.listdir(save_dir):
            if not os.path.isdir(os.path.join(save_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1
        save_dir = os.path.join(save_dir, env_name)
        save_dir += '-run{}'.format(experiment_id)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def save_model(self, step):
        if self.output is not None:
            filename = os.path.join(self.output, 'model_{}.h5'.format(step))
            self.online_net.save_model(filename)
            print('keras model written to {}'.format(filename))

    '''
    Methods subject to overloading
    '''
    def setup_algorithm(self, action_space):
        raise NotImplementedError

    def setup_nets(self, worker_dev, rep_dev, env):
        # global net
        with tf.device(rep_dev):
            if self.load_model is None:
                self.global_net = self.build_net(env)
            else:
                saved_model = self.net_cls.load_model(self.load_model)
                self.saved_weights = saved_model.get_weights()
                self.global_net = self.net_cls.from_model(saved_model)
            if self.is_master:
                self.global_net.model.summary()
            self.step_counter = StepCounter()

        # local net
        with tf.device(worker_dev):
            self.online_net = self.build_net(env)
            self.online_net.set_loss(**self.loss_kwargs)
            self.online_net.set_optimizer(**self.opt_kwargs,
                                          train_weights=self.global_net.weights)
            self.online_net.set_sync_weights(self.global_net.weights)
            self.step_counter.set_increment()

    def set_session(self, sess):
        for obj in self.global_net, self.online_net, self.step_counter:
            obj.set_session(sess)
        if self.load_model is not None:
            self.global_net.set_sync_weights(self.saved_weights)
            self.global_net.sync()

    def train_on_rollout_list(self, rollout_list):
        feed_list = [self.rollout_feed(rollout) for rollout in rollout_list]
        # concatenate individual types of feeds from the list
        train_args = map(concatenate, zip(*feed_list))
        batch_loss = self.online_net.train_on_batch(*train_args)
        return batch_loss

    def rollout_feed(self, rollout):
        raise NotImplementedError

    def rollout_target(self, rollout, value_last):
        reward_long = 0.0 if rollout.done else value_last
        r_target = zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_target[idx] = reward_long
        return r_target

