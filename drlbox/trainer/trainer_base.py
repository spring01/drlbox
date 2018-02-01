
from multiprocessing import Process, cpu_count
from .blocker import Blocker

import os
import signal
import tensorflow as tf
from drlbox.agent import AsyncRL
from .step_counter import StepCounter


class Trainer:

    KEYWORD_DICT = dict(make_env=None,
                        make_feature=None,
                        state_to_input=None,
                        load_model=None,
                        save_dir=None,              # Directory to save data to
                        num_parallel=cpu_count(),
                        dtf_port_begin=2220,
                        discount=0.99,
                        train_steps=1000000,
                        opt_learning_rate=1e-4,
                        opt_batch_size=32,
                        opt_adam_epsilon=1e-4,
                        opt_grad_clip_norm=40.0,
                        interval_sync_target=40000,
                        interval_save=10000,)

    need_target_net = False

    def __init__(self, **kwargs):
        # set default arguments
        for keyword, value in self.KEYWORD_DICT.items():
            setattr(self, keyword, value)
        # replace with user-specified arguments
        for keyword, value in kwargs.items():
            if keyword not in self.KEYWORD_DICT:
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

        self.setup_algorithm(env.action_space)

        # global net
        with tf.device(rep_dev):
            if self.load_model is None:
                global_net = self.build_net(env)
            else:
                saved_model = self.net_cls.load_model(self.load_model)
                saved_weights = saved_model.get_weights()
                global_net = self.net_cls.from_model(saved_model)
            if is_master:
                global_net.model.summary()
            step_counter = StepCounter()

        # local net
        with tf.device(worker_dev):
            online_net = self.build_net(env)
            online_net.set_loss(**self.loss_kwargs)
            online_net.set_optimizer(**self.opt_kwargs,
                                     train_weights=global_net.weights)
            online_net.set_sync_weights(global_net.weights)
            step_counter.set_increment()

        # build a separate global target net for dqn
        if self.need_target_net:
            with tf.device(rep_dev):
                target_net = self.build_net(env)
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
                            policy=self.policy,
                            rollout_builder=self.rollout_builder,
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

    def setup_algorithm(self, action_space):
        raise NotImplementedError

    def build_net(self, env):
        state, feature = self.make_feature(env.observation_space)
        return self.net_cls.from_sfa(state, feature, env.action_space)

    def get_output_folder(self, env_name):
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

