
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


class RolloutAC(Rollout):

    def get_feed(self, target_net, online_net):
        rollout_state, rollout_input, rollout_action = self.state_input_action()
        rollout_value = target_net.state_value(rollout_state)
        rollout_target = self.target(rollout_value[-1])
        rollout_adv = rollout_target - rollout_value[:-1]
        return rollout_input, rollout_action, rollout_adv, rollout_target


class RolloutMultiStepQ(Rollout):

    def get_feed(self, target_net, online_net):
        rollout_state, rollout_input, rollout_action = self.state_input_action()
        last_state = rollout_state[-1:]
        online_last_value = online_net.action_values(last_state)[-1]
        target_last_value = target_net.action_values(last_state)[-1]
        target_last_q = target_last_value[np.argmax(online_last_value)]
        rollout_target = self.target(target_last_q)
        return rollout_input, rollout_action, rollout_target

