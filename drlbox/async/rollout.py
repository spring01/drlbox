
import numpy as np


class Rollout:

    '''
    Usage: rollout = Rollout(maxlen, discount, (optional)batch_size)
        maxlen:     maximum length of a short rollout;
        discount:   discount factor gamma (for long term discounted reward).
    '''
    def __init__(self, maxlen, discount):
        self.maxlen = maxlen
        self.discount = discount

    def reset(self, state):
        self.state_list = [state]
        self.action_list = []
        self.reward_list = []
        self.done = False

    def append(self, state, action, reward, done):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.done = done

    def get_rollout_state(self):
        return np.stack(self.state_list)

    '''
    Return a tuple of bootstrapped targets
    '''
    def get_rollout_target(self, rollout_value):
        raise NotImplementedError


class RolloutAC(Rollout):

    def get_rollout_target(self, rollout_value):
        rollout_action = np.stack(self.action_list)
        reward_long = 0.0 if self.done else rollout_value[-1]
        len_batch = len(self.reward_list)
        rollout_target = np.zeros(len_batch)
        for idx in reversed(range(len_batch)):
            reward_long *= self.discount
            reward_long += self.reward_list[idx]
            rollout_target[idx] = reward_long
        rollout_adv = rollout_target - rollout_value[:-1]
        return rollout_action, rollout_adv, rollout_target


class RolloutQ(Rollout):

    def get_rollout_target(self, rollout_value):
        return rollout_value[1:]

