
import os
import numpy as np

class DQN(object):

    '''
        Add arguments into parser.
        These default values are suitable for a quick debug run.
    '''
    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--dqn_discount', default=0.99, type=float,
            help='Discount factor gamma')
        parser.add_argument('--dqn_train_steps', default=1000, type=int,
            help='Number of training sampled interactions with the environment')
        parser.add_argument('--dqn_episode_maxlen', default=100000, type=int,
            help='Maximum length of an episode')
        parser.add_argument('--dqn_batch_size', default=32, type=int,
            help='Minibatch size for Q-learning')
        parser.add_argument('--dqn_online_interval', default=4, type=int,
            help='Interval to train the online network')
        parser.add_argument('--dqn_target_interval', default=500, type=int,
            help='Interval to reset the target network')
        parser.add_argument('--dqn_print_loss_interval', default=100, type=int,
            help='Interval to print losses')
        parser.add_argument('--dqn_save_interval', default=500, type=int,
            help='Interval to save weights and memory')
        parser.add_argument('--dqn_test_interval', default=500, type=int,
            help='Evaluation interval')
        parser.add_argument('--dqn_test_episodes', default=5, type=int,
            help='Number of episodes in testing')
        parser.add_argument('--dqn_render', default='true', type=str,
            help='Do rendering or not')
        parser.add_argument('--dqn_episode_seed', default=None, type=int,
            help='Setting seed to get the same episode each time')

    def __init__(self, num_actions, q_net, memory, policy, output, args):
        self.online = q_net['online']
        self.target = q_net['target']
        self.state_to_input = q_net['interface']
        self.memory = memory
        self.policy = policy
        self.output = output
        self.dqn_discount = args.dqn_discount
        self.dqn_train_steps = args.dqn_train_steps
        self.dqn_episode_maxlen = args.dqn_episode_maxlen
        self.dqn_batch_size = args.dqn_batch_size
        self.dqn_online_interval = args.dqn_online_interval
        self.dqn_target_interval = args.dqn_target_interval
        self.dqn_print_loss_interval = args.dqn_print_loss_interval
        self.dqn_save_interval = args.dqn_save_interval
        self.dqn_test_interval = args.dqn_test_interval
        self.dqn_test_episodes = args.dqn_test_episodes
        self.dqn_render = args.dqn_render.lower() == 'true'
        self.dqn_episode_seed = args.dqn_episode_seed
        self.null_act = np.zeros([1, num_actions])
        self.null_target = np.zeros([self.dqn_batch_size, num_actions])
        self.one_hot_act = np.eye(num_actions, dtype=np.float32)

    def compile(self, loss, optimizer):
        self.online.compile(loss=loss, optimizer=optimizer)
        self.target.compile(loss=loss, optimizer=optimizer)

    def random(self, env):
        total_reward = 0.0
        for episode in range(self.dqn_test_episodes):
            episode_reward = self.run_episode(env, 'rand', 0, store=False)[0]
            print('  random episode reward: {:f}'.format(episode_reward))
            total_reward += episode_reward
        average_reward = total_reward / self.dqn_test_episodes
        print('random average episode reward: {:f}'.format(average_reward))

    def train(self, env):
        self.update_target()

        print('########## burning in some steps #############')
        while len(self.memory) < self.memory.fill:
            self.run_episode(env, mode='test', store=True)
            self.memory.print_status()

        print('########## begin training #############')
        step = 0
        episode_count = 0
        while step <= self.dqn_train_steps:
            _, step, test_flag = self.run_episode(env, 'train', step)
            episode_count += 1
            print('  iter {} out of {}'.format(step, self.dqn_train_steps))
            print('  number of episodes: {}'.format(episode_count))
            self.memory.print_status()
            if test_flag:
                print('########## testing #############')
                self.test(env)

    def test(self, env):
        total_reward = 0.0
        for episode in range(self.dqn_test_episodes):
            episode_reward = self.run_episode(env, mode='test')[0]
            print('  episode reward: {}'.format(episode_reward))
            total_reward += episode_reward
        average_reward = total_reward / self.dqn_test_episodes
        print('average episode reward: {}'.format(average_reward))

    def run_episode(self, env, mode='rand', step=0, store=False):
        print('*** New episode with mode:', mode)
        policy = self.policy[mode]

        if self.dqn_episode_seed is not None:
            env.seed(self.dqn_episode_seed)
        state = env.reset()
        episode_reward = 0.0
        test_flag = False
        for ep_iter in range(self.dqn_episode_maxlen):
            policy.update(step)
            act = self.pick_action(state, policy)
            state_next, reward, done, info = env.step(act)
            if self.dqn_render:
                env.render()
            episode_reward += reward
            reward = self.clip_reward(reward)

            if store:
                self.memory.append((state, act, reward, state_next, done))
            state = state_next

            # break if done
            if done:
                break

            # modify test_flag and step
            if mode == 'train':
                test_flag = self.extra_work_train(step) or test_flag
            step += 1
        print('*** End of episode')
        return episode_reward, step, test_flag

    def clip_reward(self, reward):
        if reward > 0.0:
            return 1.0;
        elif reward < 0.0:
            return -1.0
        else:
            return 0.0

    def extra_work_train(self, step):
        # update networks
        if _every(step, self.dqn_online_interval):
            self.memory.update_beta(step)
            self.train_online()
        if _every(step, self.dqn_target_interval):
            self.update_target()

        # save model
        if _every(step, self.dqn_save_interval):
            weights_fn = os.path.join(self.output, 'online_{}.h5'.format(step))
            print('########## saving models and memory #############')
            self.online.save_weights(weights_fn)
            print('online weights written to {}'.format(weights_fn))
            memory_fn = os.path.join(self.output, 'memory.p')
            self.memory.save(memory_fn)
            print('replay memory written to {}'.format(memory_fn))

        # print losses
        if _every(step, self.dqn_print_loss_interval):
            self.print_loss()

        # return test flag
        return _every(step, self.dqn_test_interval)

    def pick_action(self, state, policy):
        net_input = np.stack([self.state_to_input(state)])
        q_online = self.online.predict([net_input, self.null_act])[1]
        return policy.select_action(q_online)

    def train_online(self):
        batch, b_idx, b_prob, b_state, b_act, b_state_next = self.get_batch()
        batch_wts = self.memory.get_batch_weights(b_idx, b_prob)
        q_target_b, online = self.get_q_target(batch, b_state_next, b_act)
        q_online_b_act = online.predict([b_state, b_act])[0]
        self.memory.update_priority(b_idx, q_target_b - q_online_b_act)
        online.train_on_batch([b_state, b_act], [q_target_b, self.null_target],
                              sample_weight=[batch_wts, batch_wts])

    def print_loss(self):
        batch, _, _, b_state, b_act, b_state_next = self.get_batch()
        q_target_b, _ = self.get_q_target(batch, b_state_next, b_act)
        loss_online = self.online.evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        loss_target = self.target.evaluate([b_state, b_act],
            [q_target_b, self.null_target], verbose=0)
        print('losses:', loss_online[0], loss_target[0])

    def update_target(self):
        print('*** update the target network')
        self.target.set_weights(self.online.get_weights())

    def get_batch(self):
        batch, b_idx, b_prob = self.memory.sample(self.dqn_batch_size)
        b_state = []
        b_act = []
        b_state_next = []
        for st_m, act, rew, st_m_n, _ in batch:
            st = self.state_to_input(st_m)
            b_state.append(st)
            b_act.append(self.one_hot_act[act])
            st_n = self.state_to_input(st_m_n)
            b_state_next.append(st_n)
        b_state = np.stack(b_state)
        b_act = np.stack(b_act)
        b_state_next = np.stack(b_state_next)
        return batch, b_idx, b_prob, b_state, b_act, b_state_next

    def get_q_target(self, batch, b_state_next, b_act):
        if np.random.rand() < 0.5:
            online = self.target
            target = self.online
        else:
            online = self.online
            target = self.target
        q_online_b_n = online.predict([b_state_next, b_act])[1]
        q_target_b_n = target.predict([b_state_next, b_act])[1]
        q_target_b = []
        ziplist = zip(q_online_b_n, q_target_b_n, batch)
        for qon, qtn, (_, _, rew, _, db) in ziplist:
            full_reward = rew
            if not db:
                full_reward += self.dqn_discount * qon[np.argmax(qtn)]
            q_target_b.append([full_reward])
        return np.stack(q_target_b), online

def _every(step, interval):
    return not (step % interval)

