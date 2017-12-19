
import numpy as np
import pickle


FILL_PERCENT = 0.1

class ReplayMemory:

    def __init__(self, maxlen):
        self.maxlen = maxlen
        self.indices = range(self.maxlen)
        self.clear()

    def append(self, transition):
        raise NotImplementedError

    def sample(self, batch_size):
        raise NotImplementedError

    def __len__(self):
        return self.length

    def usable(self):
        return self.length >= FILL_PERCENT * self.maxlen

    def print_status(self):
        print('memory length: {}/{}'.format(self.length, self.maxlen))

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as save:
            memory = pickle.load(save)
        return memory


'''
Ring-buffer uniformly sampled replay memory.
Both `append` and `sample` are O(1)
'''
class UniformReplay(ReplayMemory):

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        idx = random.sample(self.indices, batch_size)
        return [self.ring_buffer[i] for i in idx]

    def clear(self):
        self.ring_buffer = [None] * self.maxlen
        self.index = 0
        self.length = 0



'''
Proportional prioritization implemented as a ring-buffer.
todo: change implementation to heap
'''
class PriorityReplay(ReplayMemory):

    def __init__(self, maxlen, train_steps, alpha, beta0):
        self.maxlen = maxlen
        self.alpha = alpha
        self.beta0 = beta0
        self.train_steps = float(train_steps)
        self.indices = range(self.maxlen)
        self.clear()

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.priority[self.index] = np.max(self.priority)
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def sample(self, batch_size):
        prob = self.priority / np.sum(self.priority)
        batch_idx = np.random.choice(self.indices, batch_size, False, prob)
        batch = [self.ring_buffer[i] for i in batch_idx]
        batch_prob = prob[batch_idx]
        return batch, batch_idx, batch_prob

    def update_beta(self, step_count):
        wt_end = min(step_count / self.train_steps, 1.0)
        wt_start = 1.0 - wt_end
        self.beta_annealed = self.beta0 * wt_start + 1.0 * wt_end

    def get_batch_weights(self, batch_idx, batch_prob):
        batch_weights = (self.length * batch_prob)**(-self.beta_annealed)
        batch_weights /= np.max(batch_weights)
        return batch_weights

    def update_priority(self, batch_idx, batch_td_error):
        batch_priority = np.abs(batch_td_error)
        batch_priority[batch_priority > 1.0] = 1.0
        batch_priority[batch_priority == 0.0] = 1e-16
        self.priority[batch_idx] = batch_priority**self.alpha

    def clear(self):
        self.ring_buffer = [None] * self.maxlen
        self.priority = np.zeros(self.maxlen)
        self.priority[0] = 1.0
        self.index = 0
        self.length = 0


