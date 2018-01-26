
import os
import numpy as np
from .replay import PriorityReplay

class DQN:

    def __init__(self, online, target, state_to_input,
                 replay, policy, discount, train_steps, batch_size,
                 interval_train_online, interval_sync_target, interval_save,
                 output):
        self.online = online
        self.target = target
        self.state_to_input = state_to_input
        self.replay = replay
        self.policy = policy
        self.discount = discount
        self.train_steps = train_steps
        self.batch_size = batch_size
        self.interval_train_online = interval_train_online
        self.interval_sync_target = interval_sync_target
        self.interval_save = interval_save
        self.output = output

    def train(self, env):
        self.target.sync()

        print('########## filling in replay memory #############')
        while not self.replay.usable():
            self.run_episode(env, train=False)
            self.replay.print_status()

        print('########## begin training #############')
        step = 0
        while step <= self.train_steps:
            _, step = self.run_episode(env, step, train=True)
            print('training step {}/{}'.format(step, self.train_steps))

    def run_episode(self, env, step=0, train=False):
        state = env.reset()
        episode_reward = 0.0
        while True:
            action = self.pick_action(state)
            state_next, reward, done, info = env.step(action)
            episode_reward += reward
            self.replay.append((state, action, reward, state_next, done))
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
            self.train_online()

        # sync target net
        if _every(step, self.interval_sync_target):
            self.target.sync()

        # save model and replay
        if _every(step, self.interval_save):
            output = self.output
            weights_save = os.path.join(output, 'weights_{}.p'.format(step))
            self.online.save_weights(weights_save)
            print('online net weights written to {}'.format(weights_save))
            replay_save = os.path.join(output, 'replay.p')
            self.replay.save(replay_save)
            print('replay memory written to {}'.format(replay_save))

    def train_online(self):
        batch, b_idx, b_weights = self.replay.sample(self.batch_size)
        b_state, b_act, b_q_target, abs_td, online = self.process_batch(batch)
        if type(self.replay) is PriorityReplay:
            self.replay.update_priority(b_idx, abs_td)
        online.train_on_batch(b_state, b_act, b_q_target, b_weights)

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
        b_action = []
        b_state_next = []
        for st_m, act, _, st_m_n, _ in batch:
            st = self.state_to_input(st_m)
            b_state.append(st)
            b_action.append(act)
            st_n = self.state_to_input(st_m_n)
            b_state_next.append(st_n)
        b_state = np.array(b_state, dtype=np.float32)
        b_action = np.array(b_action, dtype=np.int32)
        b_state_next = np.array(b_state_next, dtype=np.float32)

        # compute target q-values and td-error
        b_q_online = online.action_values(b_state)
        b_q_online_n = online.action_values(b_state_next)
        b_q_target_n = target.action_values(b_state_next)
        b_q_target = []
        for qon, qtn, transition in zip(b_q_online_n, b_q_target_n, batch):
            _, act, reward, _, done = transition
            full_reward = reward
            if not done:
                full_reward += self.discount * qtn[qon.argmax()]
            b_q_target.append(full_reward)
        b_q_target = np.array(b_q_target, dtype=np.float32)
        b_q_online_a = np.array([o[a] for o, a in zip(b_q_online, b_action)])
        abs_td_error = np.abs(b_q_target - b_q_online_a)
        return b_state, b_action, b_q_target, abs_td_error, online

def _every(step, interval):
    return not (step % interval)

