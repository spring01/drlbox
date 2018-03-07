
from multiprocessing import Process, Event, cpu_count
import socket

import os
import signal
import time
from datetime import timedelta

import tensorflow as tf
import numpy as np
from drlbox.layer.noisy_dense import NoisyDenseIG
from drlbox.net.kfac import KfacOptimizerTV, build_layer_collection
from drlbox.common.tasker import Tasker
from .step_counter import StepCounter
from .rollout import Rollout


LOCALHOST = 'localhost'
JOBNAME = 'local'

class Trainer(Tasker):

    KEYWORD_DICT = {**Tasker.KEYWORD_DICT,
                    **dict(feature_maker=None,
                           save_dir=None,           # Directory to save data to
                           num_parallel=cpu_count(),
                           port_begin=2220,
                           discount=0.99,
                           train_steps=1000000,
                           opt_learning_rate=1e-4,
                           opt_batch_size=32,
                           opt_type='adam',         # 'adam' or 'kfac'
                           opt_adam_epsilon=1e-4,
                           opt_clip_norm=40.0,
                           kfac_cov_ema_decay=0.95,
                           kfac_damping=1e-3,
                           kfac_trust_radius=1e-3,
                           kfac_inv_upd_interval=10,
                           noisynet=None,           # None, 'ig', or 'fg'
                           interval_save=10000,
                           catch_signal=False,
                           )}

    def run(self):
        self.port_list = [self.port_begin + i for i in range(self.num_parallel)]
        for port in self.port_list:
            if not self.port_available(LOCALHOST, port):
                raise NameError('port {} is not available'.format(port))
        self.print('Claiming {} port {} ...'.format(LOCALHOST, self.port_list))
        self.event_finished = Event()
        self.worker_list = []
        try:
            for wid in range(self.num_parallel):
                worker = Process(target=self.worker, args=(wid,))
                worker.start()
                self.worker_list.append(worker)
        except:
            self.terminate_workers()

        # set handlers if requested
        if self.catch_signal:
            self.default_sigint_handler = signal.signal(signal.SIGINT,
                                                        self.signal_handler)
            self.default_sigterm_handler = signal.signal(signal.SIGTERM,
                                                         self.signal_handler)
            self.print('SIGINT and SIGTERM will be caught by drlbox')

        # terminates the entire training when the master worker terminates
        master_worker = self.worker_list[0]
        wait_counter = 0
        start_time = time.time()
        while not self.event_finished.is_set():
            wait_counter += 1
            if wait_counter >= 3000:
                wait_counter = 0
                elapsed = int(time.time() - start_time)
                time_str = str(timedelta(seconds=elapsed))
                self.print('Elapsed time:', time_str)
            time.sleep(0.1)
        self.print('A worker just terminated -- training should end soon')
        self.terminate_workers()

        # restore default handlers
        if self.catch_signal:
            signal.signal(signal.SIGINT, self.default_sigint_handler)
            signal.signal(signal.SIGTERM, self.default_sigterm_handler)
            self.print('SIGINT and SIGTERM default handlers reset to default')
        self.print('Asynchronous training has ended')

    def signal_handler(self, signum, frame):
        self.event_finished.set()

    def terminate_workers(self):
        for worker in self.worker_list[::-1]:
            while worker.is_alive():
                worker.terminate()
                time.sleep(0.01)

    def worker(self, wid):
        env = self.env_maker()
        self.is_master = wid == 0
        if self.is_master and self.save_dir is not None:
            env_name = 'UnknownEnv-v0' if env.spec is None else env.spec.id
            self.output = self.get_output_dir(env_name)
        else:
            self.output = None

        # ports, cluster, and server
        cluster_list = ['{}:{}'.format(LOCALHOST, p) for p in self.port_list]
        cluster = tf.train.ClusterSpec({JOBNAME: cluster_list})
        server = tf.train.Server(cluster, job_name=JOBNAME, task_index=wid)
        self.print('Starting server #{}'.format(wid))

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

        self.event_finished.set()
        if self.is_master:
            while True:
                time.sleep(1)

    def train_on_env(self, env):
        step = self.step_counter.step_count()
        if self.is_master:
            last_save = step
            self.save_model(step)

        state = env.reset()
        episode_reward = 0.0
        while step <= self.train_steps:
            self.online_net.sync()
            if self.noisynet is not None:
                self.online_net.sample_noise()
            rollout_list = [Rollout(state)]
            for batch_step in range(self.opt_batch_size):
                net_input = self.state_to_input(state)
                act_val = self.online_net.action_values([net_input])[0]
                action = self.policy.select_action(act_val)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                rollout_list[-1].append(state, action, reward, done, act_val)
                if done:
                    state = env.reset()
                    if batch_step < self.opt_batch_size - 1:
                        rollout_list.append(Rollout(state))
                    self.print('episode reward {:5.2f}'.format(episode_reward))
                    episode_reward = 0.0

            batch_loss = self.train_on_rollout_list(rollout_list)

            self.step_counter.increment(self.opt_batch_size)
            step = self.step_counter.step_count()
            if self.is_master:
                if step - last_save > self.interval_save:
                    self.save_model(step)
                    last_save = step
                str_step = 'training step {}/{}'.format(step, self.train_steps)
                self.print(str_step + ', loss {:3.3f}'.format(batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_model(step)

    def port_available(self, host, port):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        return sock.connect_ex((host, port)) != 0

    def build_net(self, env=None, is_global=False):
        net = self.net_cls()
        if self.noisynet == 'ig':
            net.dense_layer = NoisyDenseIG
            self.print('Using independent Gaussian NoisyNet')
        if self.load_model is not None and is_global:
            model = self.do_load_model()
            self.saved_weights = model.get_weights()
        else:
            state, feature = self.feature_maker(env.observation_space)
            model = net.build_model(state, feature, env.action_space)
        net.set_model(model)
        if self.noisynet is not None:
            net.set_noise_list()
        return net

    def set_online_optimizer(self):
        if self.opt_type == 'adam':
            adam = tf.train.AdamOptimizer(self.opt_learning_rate,
                                          epsilon=self.opt_adam_epsilon)
            self.online_net.set_optimizer(adam, clip_norm=self.opt_clip_norm,
                                          train_weights=self.global_net.weights)
        elif self.opt_type == 'kfac':
            layer_list = self.online_net.model.layers
            layer_collection = build_layer_collection(layer_list,
                self.online_net.kfac_loss_list)
            kfac = KfacOptimizerTV(learning_rate=self.opt_learning_rate,
                                   cov_ema_decay=self.kfac_cov_ema_decay,
                                   damping=self.kfac_damping,
                                   norm_constraint=self.kfac_trust_radius,
                                   layer_collection=layer_collection,
                                   var_list=self.online_net.weights)
            self.online_net.set_kfac(kfac, self.kfac_inv_upd_interval,
                                     train_weights=self.global_net.weights)
        else:
            raise ValueError('Optimizer type {} invalid'.format(self.opt_type))

    def get_output_dir(self, env_name):
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir, exist_ok=True)
            self.print('Made output dir', self.save_dir)
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
            self.print('keras model written to {}'.format(filename))

    '''
    Methods subject to overloading
    '''
    def setup_algorithm(self, action_space):
        raise NotImplementedError

    def setup_nets(self, worker_dev, rep_dev, env):
        # global net
        with tf.device(rep_dev):
            self.global_net = self.build_net(env, is_global=True)
            if self.is_master and self.verbose:
                self.global_net.model.summary()
            self.step_counter = StepCounter()

        # local net
        with tf.device(worker_dev):
            self.online_net = self.build_net(env)
            self.online_net.set_loss(**self.loss_kwargs)
            self.set_online_optimizer()
            self.online_net.set_sync_weights(self.global_net.weights)
            self.step_counter.set_increment()

    def set_session(self, sess):
        for obj in self.global_net, self.online_net, self.step_counter:
            obj.set_session(sess)
        if self.load_model is not None:
            self.global_net.set_sync_weights(self.saved_weights)
            self.global_net.sync()

    def train_on_rollout_list(self, rollout_list):
        rl_state = []
        rl_slice = []
        last_index = 0
        for rollout in rollout_list:
            r_state = []
            for state in rollout.state_list:
                r_state.append(self.state_to_input(state))
            r_state = np.array(r_state)
            rl_state.append(r_state)
            index = last_index + len(r_state)
            rl_slice.append(slice(last_index, index))
            last_index = index
        cc_state = np.concatenate(rl_state)

        # cc_boots is a tuple of concatenated bootstrap quantities
        cc_boots = self.rollout_list_bootstrap(cc_state, rl_slice)

        # rl_boots is a list of tuple of boostrap quantities
        # and each tuple corresponds to a rollout
        rl_boots = [tuple(boot[r_slice] for boot in cc_boots)
                    for r_slice in rl_slice]

        # feed_list contains all arguments to train_on_batch
        feed_list = []
        for rollout, r_state, r_boot in zip(rollout_list, rl_state, rl_boots):
            r_input = r_state[:-1]
            r_feeds = self.rollout_feed(rollout, *r_boot)
            feed_list.append((r_input, *r_feeds))

        # concatenate individual types of feeds from the list
        train_args = map(np.concatenate, zip(*feed_list))
        batch_loss = self.online_net.train_on_batch(*train_args)
        return batch_loss

    def rollout_list_bootstrap(self, cc_state):
        raise NotImplementedError

    def rollout_feed(self, rollout, *rollout_bootstraps):
        raise NotImplementedError

    def rollout_target(self, rollout, value_last):
        reward_long = 0.0 if rollout.done else value_last
        r_target = np.zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_target[idx] = reward_long
        return r_target

