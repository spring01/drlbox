

class Rollout:

    '''
    Usage: rollout = Rollout(state)
    '''
    def __init__(self, state):
        self.state_list = [state]
        self.action_list = []
        self.reward_list = []
        self.act_val_list = []
        self.done = False

    def append(self, state, action, reward, done, act_val):
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        self.act_val_list.append(act_val)
        self.done = done

    '''
    Length of the rollout is (number of states - 1) or (number of rewards)
    '''
    def __len__(self):
        return len(self.reward_list)

