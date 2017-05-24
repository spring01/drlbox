
import numpy as np
import cPickle as pickle


class PriorityMemory(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--memory_maxlen', default=100000, type=int,
            help='Replay memory length')
        parser.add_argument('--memory_fill', default=10000, type=int,
            help='Fill the replay memory to how much length before update')
        parser.add_argument('--memory_alpha', default=0.6, type=float,
            help='Exponent alpha in prioritized replay memory')
        parser.add_argument('--memory_beta0', default=0.4, type=float,
            help='Initial beta in prioritized replay memory')

    def __init__(self, train_steps, args):
        self.maxlen = args.memory_maxlen
        self.fill = args.memory_fill
        self.alpha = args.memory_alpha
        self.beta0 = args.memory_beta0
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

    def get_batch_weights(self, batch_idx, batch_prob, iter_num):
        wt_end = min(iter_num / self.train_steps, 1.0)
        wt_start = 1.0 - wt_end
        beta_annealed = self.beta0 * wt_start + 1.0 * wt_end
        batch_weights = (self.length * batch_prob)**(-beta_annealed)
        batch_weights /= np.max(batch_weights)
        return batch_weights

    def update_priority(self, batch_idx, batch_td_error):
        batch_priority = np.abs(batch_td_error)
        batch_priority[batch_priority > 1.0] = 1.0
        batch_priority[batch_priority == 0.0] = 1e-16
        self.priority[batch_idx] = batch_priority**self.alpha

    def clear(self):
        self.ring_buffer = [None for _ in xrange(self.maxlen)]
        self.priority = np.zeros(self.maxlen)
        self.priority[0] = 1.0
        self.index = 0
        self.length = 0

    def __len__(self):
        return self.length

    def print_status(self):
        print '  memory length: {:d}/{:d}'.format(self.length, self.maxlen)

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as save:
            memory = pickle.load(save)
        self.ring_buffer[:memory.length] = memory.ring_buffer[:memory.length]
        self.priority[:memory.length] = memory.priority[:memory.length]
        self.index = memory.length % self.maxlen
        self.length = memory.length

