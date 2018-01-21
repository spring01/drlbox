
import os
import numpy as np


DUMMY_LOSS = 9999.0

class AsyncRL:

    '''
    `online_net` needs to have its weight-sync operation set to the global net
    by calling `set_sync_weights`;
    `target_net` is either a reference to `online_net` (in actor-critic)
    or has its weight-sync operation set to the global online net;
    '''
    def __init__(self, is_master, online_net, target_net, state_to_input,
                 policy, rollout, batch_size, train_steps, step_counter,
                 interval_sync_target, interval_save, output):
        self.is_master = is_master
        self.online_net = online_net
        self.target_net = target_net
        self.state_to_input = state_to_input
        self.policy = policy
        self.rollout = rollout
        self.batch_size = batch_size
        self.train_steps = train_steps
        self.step_counter = step_counter
        self.interval_sync_target = interval_sync_target
        self.interval_save = interval_save
        self.output = output

        # Cache to concatenate rollouts into (possibly larger) batches
        self.batch_cache = []
        self.batch_cache_size = 0

        # Initialize batch_loss to a dummy value
        self.batch_loss = DUMMY_LOSS

    def train(self, env):
        rollout = self.rollout
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
            rollout.reset(state)
            for t in range(rollout.maxlen):
                action_values = self.online_net.action_values([state])[0]
                action = self.policy.select_action(action_values)
                state, reward, done, info = env.step(action)
                episode_reward += reward
                state = self.state_to_input(state)
                rollout.append(state, action, reward, done)
                if done:
                    state = env.reset()
                    state = self.state_to_input(state)
                    print('episode reward {:5.2f}'.format(episode_reward),
                          flush=True)
                    episode_reward = 0.0
                    break

            '''
            The idea is to cache rollouts until cache size exceeds batch size
            and then the net is trained with a batch of size exactly batch size.
            The remaining leftover part is then put back into `batch_cache`.
            '''
            rollout_feed = rollout.get_feed(self.target_net, self.online_net)

            '''
            For DQN:
                `rollout_feed` is a tuple `(inputs, targets)` where
                    both `inputs` and `targets` are of length `len(rollout)`.
                `train_args` is `(batch_inputs, batch_targets)`;
                `leftovers` is `(leftover_inputs, leftover_targets)`.
            For actor-critic:
                `rollout_feed` is `(inputs, actions, advantages, targets)` where
                    anyone is of length `len(rollout)`.
                `train_args` and `leftovers` are the corresponding batch version
            '''
            self.batch_cache.append(rollout_feed)
            self.batch_cache_size += len(rollout)
            if self.batch_cache_size >= self.batch_size:
                train_left = map(self.train_leftover, zip(*self.batch_cache))
                train_args, leftovers = zip(*train_left)
                self.batch_cache = [leftovers]
                self.batch_cache_size = len(leftovers[0])
                self.batch_loss = self.online_net.train_on_batch(*train_args)

            step_counter.increment(t)
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
                print(str_step + ', loss {:3.3f}'.format(self.batch_loss))
        # save at the end of training
        if self.is_master:
            self.save_weights(step)

    '''
    bc_quantity is of size >= self.batch_size; this function
    Splits bc_quantity into a training batch of size exactly self.batch_size
    and a leftover batch of the remaining size.
    Argument:
        bc_quantity: a list of size >= self.batch_size;
    '''
    def train_leftover(self, bc_quantity):
        bc_quantity = np.concatenate(bc_quantity)
        train = bc_quantity[:self.batch_size]
        leftover = bc_quantity[self.batch_size:]
        return train, leftover

    def save_weights(self, step):
        weights_save = os.path.join(self.output, 'weights_{}.p'.format(step))
        self.online_net.save_weights(weights_save)
        print('global net weights written to {}'.format(weights_save))


