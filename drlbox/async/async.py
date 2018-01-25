
import os
import builtins
import numpy as np


print = lambda *args, **kwargs: builtins.print(*args, **kwargs, flush=True)

class AsyncRL:

    '''
    `online_net` needs to have its weight-sync operation set to the global net
    by calling `set_sync_weights`;
    `target_net` is either a reference to `online_net` (in actor-critic)
    or has its weight-sync operation set to the global online net;
    '''
    def __init__(self, is_master, online_net, target_net, state_to_input,
                 policy, rollout_builder, batch_size, train_steps, step_counter,
                 interval_sync_target, interval_save, output):
        self.is_master = is_master
        self.online_net = online_net
        self.target_net = target_net
        self.state_to_input = state_to_input
        self.policy = policy
        self.rollout_builder = rollout_builder
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.step_counter = step_counter
        self.interval_sync_target = interval_sync_target
        self.interval_save = interval_save
        self.output = output

    def train(self, env):
        step_counter = self.step_counter
        step = step_counter.step_count()
        if self.is_master:
            last_save = step
            last_sync_target = step
            self.save_weights(step)

        state = env.reset()
        state = self.state_to_input(state)
        episode_reward = 0.0
        while step <= self.train_steps:
            self.online_net.sync()
            rollout_list = [self.rollout_builder(state)]
            for batch_step in range(self.batch_size):
                action_values = self.online_net.action_values([state])[0]
                action = self.policy.select_action(action_values)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                state = self.state_to_input(state)
                rollout_list[-1].append(state, action, reward, done)
                if done:
                    state = env.reset()
                    state = self.state_to_input(state)
                    if batch_step < self.batch_size - 1:
                        rollout_list.append(self.rollout_builder(state))
                    print('episode reward {:5.2f}'.format(episode_reward))
                    episode_reward = 0.0

            '''
            feed_list is a list of tuples:
            (inputs, actions, advantages, targets) for actor-critic;
            (inputs, targets) for dqn.
            '''
            feed_list = [rollout.get_feed(self.target_net, self.online_net)
                         for rollout in rollout_list]

            # concatenate individual types of feeds from the list
            train_args = map(np.concatenate, zip(*feed_list))
            batch_loss = self.online_net.train_on_batch(*train_args)

            step_counter.increment(self.batch_size)
            step = step_counter.step_count()
            if self.is_master:
                if step - last_save > self.interval_save:
                    self.save_weights(step)
                    last_save = step
                if self.target_net is not self.online_net:
                    if step - last_sync_target > self.interval_sync_target:
                        self.target_net.sync()
                        last_sync_target = step
                str_step = 'training step {}/{}'.format(step, self.train_steps)
                print(str_step + ', loss {:3.3f}'.format(batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_weights(step)

    def save_weights(self, step):
        weights_save = os.path.join(self.output, 'weights_{}.p'.format(step))
        self.online_net.save_weights(weights_save)
        print('global net weights written to {}'.format(weights_save))


