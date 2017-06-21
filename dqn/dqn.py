
import os
import numpy as np

class DQN(object):

    episode_maxlen = 100000
    batch_size     = 32

    def __init__(self, online, target, state_to_input,
                 memory, policy, discount, train_steps,
                 interval_train_online, interval_sync_target, interval_save,
                 output):
        self.online = online
        self.target = target
        self.state_to_input = state_to_input
        self.memory = memory
        self.policy = policy
        self.discount = discount
        self.train_steps = train_steps
        self.interval_train_online = interval_train_online
        self.interval_sync_target = interval_sync_target
        self.interval_save = interval_save
        self.output = output

    def train(self, env):
        self.target.sync()

        print('########## filling in memory #############')
        while len(self.memory) < self.memory.fill:
            self.run_episode(env, train=False)
            self.memory.print_status()

        print('########## begin training #############')
        step = 0
        while step <= self.train_steps:
            self.policy.update(step)
            _, step = self.run_episode(env, step, train=True)
            print('training step {}/{}'.format(step, self.train_steps))

    def run_episode(self, env, step=0, train=False):
        state = env.reset()
        episode_reward = 0.0
        for _ in range(self.episode_maxlen):
            action = self.pick_action(state)
            state_next, reward, done, info = env.step(action)
            episode_reward += reward
            self.memory.append((state, action, reward, state_next, done))
            state = state_next

            # break if done
            if done:
                break

            if train:
                self.extra_work_train(step)
            step += 1
        return episode_reward, step

    def pick_action(self, state):
        net_input = np.stack([self.state_to_input(state)])
        q_online = self.online.action_values(net_input)[0]
        return self.policy.select_action(q_online)

    def extra_work_train(self, step):
        # train online net
        if _every(step, self.interval_train_online):
            self.memory.update_beta(step)
            self.train_online()

        # sync target net
        if _every(step, self.interval_sync_target):
            self.target.sync()

        # save model
        if _every(step, self.interval_save):
            output = self.output
            weights_save = os.path.join(output, 'weights_{}.p'.format(step))
            self.online.save_weights(weights_save)
            print('online net weights written to {}'.format(weights_save))
            memory_save = os.path.join(output, 'memory.p')
            self.memory.save(memory_save)
            print('replay memory written to {}'.format(memory_save))

    def train_online(self):
        batch, b_idx, b_prob = self.memory.sample(self.batch_size)
        b_weights = self.memory.get_batch_weights(b_idx, b_prob)
        b_state, b_q_target, td_error, online = self.process_batch(batch)
        self.memory.update_priority(b_idx, td_error)
        online.train_on_batch(b_state, b_q_target, sample_weight=b_weights)

    def process_batch(self, batch):
        # roll online/target nets for double q-learning
        if np.random.rand() < 0.5:
            online = self.target
            target = self.online
        else:
            online = self.online
            target = self.target

        # build stacked batch of states and batch of next states
        b_state = []
        b_state_next = []
        for st_m, act, _, st_m_n, _ in batch:
            st = self.state_to_input(st_m)
            b_state.append(st)
            st_n = self.state_to_input(st_m_n)
            b_state_next.append(st_n)
        b_state = np.stack(b_state)
        b_state_next = np.stack(b_state_next)

        # compute target q-values and td-error
        b_q_online = online.action_values(b_state)
        b_q_online_n = online.action_values(b_state_next)
        b_q_target_n = target.action_values(b_state_next)
        b_q_target = b_q_online.copy()
        ziplist = zip(b_q_target, b_q_online_n, b_q_target_n, batch)
        for qt, qon, qtn, trans in ziplist:
            _, act, reward, _, done = trans
            full_reward = reward
            if not done:
                full_reward += self.discount * qtn[qon.argmax()]
            qt[act] = full_reward
        td_error = np.sum(b_q_target - b_q_online, axis=1)
        return b_state, b_q_target, td_error, online

def _every(step, interval):
    return not (step % interval)

