
import numpy as np


class Rollout:

    '''
    Usage: rollout = Rollout(state, discount)
        discount:   discount factor gamma (for long term discounted reward);
    '''
    def __init__(self, state, discount):
        self.discount = discount
        self.state_list = [state]
        self.action_list = []
        self.reward_list = []
        self.done = False

    def append(self, state, action, reward, done):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.done = done

    '''
    Length of the rollout is (number of states - 1) or (number of rewards)
    '''
    def __len__(self):
        return len(self.reward_list)

    '''
    target_net: target network for calculating the update target;
    online_net: in double Q learning, the online network is used to find out
                the optimal action for the bootstrapped Q.
    '''
    def get_feed(self, target_net, online_net):
        raise NotImplementedError

    def state_input_action(self):
        rollout_state = np.stack(self.state_list)
        rollout_input = rollout_state[:-1]
        rollout_action = np.stack(self.action_list)
        return rollout_state, rollout_input, rollout_action

    def target(self, value_last):
        reward_long = 0.0 if self.done else value_last
        rollout_target = np.zeros(len(self))
        for idx in reversed(range(len(self))):
            reward_long *= self.discount
            reward_long += self.reward_list[idx]
            rollout_target[idx] = reward_long
        return rollout_target

