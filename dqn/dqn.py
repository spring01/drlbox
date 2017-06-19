
import os
import numpy as np

class DQN(object):

    episode_maxlen        = 100000
    batch_size            = 32
    train_online_interval = 4
    sync_target_interval  = 40000
    save_interval         = 40000

    def __init__(self, num_actions, online, target, state_to_input,
                 output, memory, policy, discount, train_steps):
        self.online = online
        self.target = target
        self.state_to_input = state_to_input
        self.output = output
        self.memory = memory
        self.policy = policy
        self.discount = discount
        self.train_steps = train_steps

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
            print('training step {} out of {}'.format(step, self.train_steps))
            self.memory.print_status()

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

            # modify test_flag and step
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
        if _every(step, self.train_online_interval):
            self.memory.update_beta(step)
            self.train_online()

        # sync target net
        if _every(step, self.sync_target_interval):
            self.target.sync()

        # save model
        if _every(step, self.save_interval):
            output = self.output
            weights_save = os.path.join(output, 'weights_{}.p'.format(step))
            self.online.save_weights(weights_save)
            print('online net weights written to {}'.format(weights_save))
            memory_save = os.path.join(output, 'memory.p')
            self.memory.save(memory_save)
            print('replay memory written to {}'.format(memory_save))

    def train_online(self):
        batch, b_idx, b_prob = self.memory.sample(self.batch_size)
        batch_weights = self.memory.get_batch_weights(b_idx, b_prob)
        b_state, q_target_b, td_error, online = self.process_batch(batch)
        self.memory.update_priority(b_idx, td_error)
        online.train_on_batch(b_state, q_target_b, sample_weight=batch_weights)

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
        for st_m, act, rew, st_m_n, _ in batch:
            st = self.state_to_input(st_m)
            b_state.append(st)
            st_n = self.state_to_input(st_m_n)
            b_state_next.append(st_n)
        b_state = np.stack(b_state)
        b_state_next = np.stack(b_state_next)

        # compute target q-values and td-error
        q_online_b = online.action_values(b_state)
        q_online_b_n = online.action_values(b_state_next)
        q_target_b_n = target.action_values(b_state_next)
        q_target_b = q_online_b.copy()
        ziplist = zip(q_target_b, q_online_b_n, q_target_b_n, batch)
        for qt, qon, qtn, trans in ziplist:
            _, act, reward, _, done = trans
            full_reward = reward
            if not done:
                full_reward += self.discount * qon[np.argmax(qtn)]
            qt[act] = full_reward
        td_error = np.sum(q_target_b - q_online_b, axis=1)
        return b_state, q_target_b, td_error, online

def _every(step, interval):
    return not (step % interval)

