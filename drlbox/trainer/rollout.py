
import numpy as np


class Rollout:

    '''
    Usage: rollout = Rollout(state)
    '''
    def __init__(self, state):
        self.state_list = [state]
        self.action_list = []
        self.reward_list = []
        self.action_val_list = []
        self.done = False

    def append(self, state, action, reward, done, action_val):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.action_val_list.append(action_val)
        self.done = done

    '''
    Length of the rollout is (number of states - 1) or (number of rewards)
    '''
    def __len__(self):
        return len(self.reward_list)

    def state_input_action(self):
        rollout_state = np.stack(self.state_list)
        rollout_input = rollout_state[:-1]
        rollout_action = np.stack(self.action_list)
        return rollout_state, rollout_input, rollout_action

    def action_val(self):
        return np.stack(self.action_val_list)
