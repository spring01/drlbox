
import gym.spaces

from multiprocessing import Process, cpu_count
from drlbox.async.blocker import Blocker

import os
import signal
import tensorflow as tf
from drlbox.async.async import AsyncRL
from drlbox.net import QNet, ACNet, ACKTRNet, NoisyQNet, NoisyACNet
from drlbox.net.kfac import KfacOptimizerTV
from drlbox.async.rollout import RolloutAC, RolloutMultiStepQ
from drlbox.async.step_counter import StepCounter
from drlbox.common.policy import StochasticDisc, StochasticCont, DecayEpsGreedy


KEYWORD_DICT = dict(make_env=None,
                    make_feature=None,
                    state_to_input=None,
                    algorithm='a3c',
                    load_model=None,
                    save_dir=None,                  # Directory to save data to
                    num_parallel=cpu_count(),
                    dtf_port_begin=2220,
                    discount=0.99,
                    a3c_entropy_weight=1e-2,
                    policy_eps_start=1.0,
                    policy_eps_end=0.01,
                    policy_eps_decay_steps=1000000,
                    policy_sto_cont_min_var=1e-4,
                    train_steps=1000000,
                    opt_learning_rate=1e-4,
                    opt_batch_size=32,
                    opt_adam_epsilon=1e-4,
                    opt_grad_clip_norm=40.0,
                    kfac_cov_ema_decay=0.95,
                    kfac_damping=1e-3,
                    kfac_trust_radius=1e-3,
                    kfac_inv_upd_interval=10,
                    interval_sync_target=40000,
                    interval_save=10000,)

class Trainer:

    def __init__(self, **kwargs):
        # set default arguments
        for keyword, value in KEYWORD_DICT.items():
            setattr(self, keyword, value)
        # replace with user-specified arguments
        for keyword, value in kwargs.items():
            if keyword not in KEYWORD_DICT:
                raise ValueError('Argument "{}" not valid'.format(keyword))
            setattr(self, keyword, value)

    def run(self):
        worker_list = []
        for wid in range(self.num_parallel):
            worker = Process(target=self.worker, args=(wid,))
            worker.start()
            worker_list.append(worker)
        Blocker().block()
        for worker in worker_list:
            worker.terminate()
        print('AsyncRL training ends')

    def worker(self, wid):
        env, env_name = self.make_env()

        # ports, cluster, and server
        port_list = [self.dtf_port_begin + i for i in range(self.num_parallel)]
        is_master = wid == 0
        this_port = self.dtf_port_begin + wid
        cluster_list = ['localhost:{}'.format(port) for port in port_list]
        cluster = tf.train.ClusterSpec({'local': cluster_list})
        server = tf.train.Server(cluster, job_name='local', task_index=wid)
        print('Starting server #{}'.format(wid))

        # global/local actor-critic nets
        worker_dev = '/job:local/task:{}/cpu:0'.format(wid)
        rep_dev = tf.train.replica_device_setter(worker_device=worker_dev,
                                                 cluster=cluster)

        (net_cls, loss_kwargs, opt_kwargs, rollout_builder,
         policy) = self.select_algorithm(env.action_space)

        # global net
        with tf.device(rep_dev):
            if self.load_model is None:
                global_net = self.build_net(net_cls, env)
            else:
                saved_model = net_cls.load_model(self.load_model)
                saved_weights = saved_model.get_weights()
                global_net = net_cls.from_model(saved_model)
            if is_master:
                global_net.model.summary()
            step_counter = StepCounter()

        # local net
        with tf.device(worker_dev):
            online_net = self.build_net(net_cls, env)
            online_net.set_loss(**loss_kwargs)
            online_net.set_optimizer(**opt_kwargs,
                                     train_weights=global_net.weights)
            online_net.set_sync_weights(global_net.weights)
            step_counter.set_increment()

        # build a separate global target net for dqn
        if self.need_target_net():
            with tf.device(rep_dev):
                target_net = self.build_net(net_cls, env)
                target_net.set_sync_weights(global_net.weights)
        else: # make target net a reference to the local net
            target_net = online_net

        # begin tensorflow session, build async RL agent and train
        with tf.Session('grpc://localhost:{}'.format(this_port)) as sess:
            sess.run(tf.global_variables_initializer())
            for obj in global_net, online_net, step_counter:
                obj.set_session(sess)
            if target_net is not online_net:
                target_net.set_session(sess)
            output = self.get_output_folder(env_name) if is_master else None
            agent = AsyncRL(is_master=is_master,
                            online_net=online_net, target_net=target_net,
                            state_to_input=self.state_to_input,
                            policy=policy, rollout_builder=rollout_builder,
                            batch_size=self.opt_batch_size,
                            train_steps=self.train_steps,
                            step_counter=step_counter,
                            interval_sync_target=self.interval_sync_target,
                            interval_save=self.interval_save,
                            output=output)
            if self.load_model is not None:
                global_net.set_sync_weights(saved_weights)
                global_net.sync()

            # train the agent
            agent.train(env)

            # terminates the entire training when the master worker terminates
            if is_master:
                print('Master worker terminates')
                os.kill(os.getppid(), signal.SIGTERM)

    def select_algorithm(self, action_space):
         # algorithms differ in terms of network structure, rollout, and policy
        if self.algorithm in {'a3c', 'acktr', 'a3c-noisynet'}:
            loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                               min_var=self.policy_sto_cont_min_var)
            if self.algorithm == 'a3c':
                net_cls = ACNet
            elif self.algorithm == 'a3c-noisynet':
                net_cls = NoisyACNet
            elif self.algorithm == 'acktr':
                net_cls = ACKTRNet

            adam_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)

            if self.algorithm in {'a3c', 'a3c-noisynet'}:
                opt_kwargs = adam_kwargs
            elif self.algorithm == 'acktr':
                opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                                  cov_ema_decay=self.kfac_cov_ema_decay,
                                  damping=self.kfac_damping,
                                  trust_radius=self.kfac_trust_radius,
                                  inv_upd_interval=self.kfac_inv_upd_interval)

            # rollout
            rollout_builder = lambda s: RolloutAC(s, self.discount)

            # policy
            if discrete_action(action_space):
                policy = StochasticDisc()
            elif continuous_action(action_space):
                policy = StochasticCont(low=action_space.low,
                                        high=action_space.high,
                                        min_var=self.policy_sto_cont_min_var)
        elif self.algorithm in {'dqn', 'dqn-noisynet'}:
            loss_kwargs = {}
            if self.algorithm == 'dqn':
                net_cls = QNet
            elif self.algorithm == 'dqn-noisynet':
                net_cls = NoisyQNet
            opt_kwargs = adam_kwargs

            # rollout
            rollout_builder = lambda s: RolloutMultiStepQ(s, self.discount)

            # policy
            eps_start = self.policy_eps_start
            eps_end = self.policy_eps_end
            eps_delta = (eps_start - eps_end) / self.policy_eps_decay_steps
            policy = DecayEpsGreedy(eps_start, eps_end, eps_delta)
        else:
            raise ValueError('Algorithm "{}" not valid.'.format(self.algorithm))
        return net_cls, loss_kwargs, opt_kwargs, rollout_builder, policy

    def build_net(self, net_cls, env):
        state, feature = self.make_feature(env.observation_space)
        return net_cls.from_sfa(state, feature, env.action_space)

    def need_target_net(self):
        return self.algorithm in {'dqn', 'dqn-noisynet'}

    def get_output_folder(self, env_name):
        if self.save_dir is None:
            return None
        parent_dir = self.save_dir
        experiment_id = 0
        if not os.path.isdir(parent_dir):
            subprocess.call(['mkdir', '-p', parent_dir])
            print('Made output dir', parent_dir)
        for folder_name in os.listdir(parent_dir):
            if not os.path.isdir(os.path.join(parent_dir, folder_name)):
                continue
            try:
                folder_name = int(folder_name.split('-run')[-1])
                if folder_name > experiment_id:
                    experiment_id = folder_name
            except:
                pass
        experiment_id += 1

        parent_dir = os.path.join(parent_dir, env_name)
        parent_dir += '-run{}'.format(experiment_id)
        subprocess.call(['mkdir', '-p', parent_dir])
        return parent_dir

def discrete_action(action_space):
    return type(action_space) is gym.spaces.discrete.Discrete

def continuous_action(action_space):
    return type(action_space) is gym.spaces.box.Box




