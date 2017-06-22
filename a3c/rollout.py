
import numpy as np


class Rollout(object):

    def __init__(self, maxlen, num_actions):
        self.action_1h_converter = np.eye(num_actions)
        self.maxlen = maxlen

    def reset(self, state):
        self.state_list = [state]
        self.action_1h_list = []
        self.reward_list = []
        self.done = False

    def append(self, state, action, reward, done):
        self.state_list.append(state)
        action_1h = self.action_1h_converter[action]
        self.action_1h_list.append(action_1h)
        self.reward_list.append(reward)
        self.done = done

    def get_batch_state(self):
        return np.stack(self.state_list)

    def get_batch_target(self, batch_value):
        batch_action_1h = np.stack(self.action_1h_list)
        reward_long = 0.0 if self.done else batch_value[-1]
        reward_long_list = []
        for reward in reversed(self.reward_list):
            reward_long = reward + 0.99 * reward_long
            reward_long_list.append(reward_long)
        batch_target = np.stack(reversed(reward_long_list))
        batch_adv = batch_target - batch_value[:-1]
        return batch_action_1h, batch_adv, batch_target

