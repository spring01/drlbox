
import numpy as np
import cPickle as pickle


class PriorityMemory(object):

    @staticmethod
    def add_arguments(parser):
        parser.add_argument('--memory_size', default=100000, type=int,
            help='Replay memory size')
        parser.add_argument('--memory_fill', default=10000, type=int,
            help='Fill the replay memory to how much size before update')
        parser.add_argument('--memory_alpha', default=0.6, type=float,
            help='Exponent alpha in prioritized replay memory')
        parser.add_argument('--memory_beta0', default=0.4, type=float,
            help='Initial beta in prioritized replay memory')

    def __init__(self, train_steps, args):
        self.memory_size = args.memory_size
        self.memory_fill = args.memory_fill
        self.memory_alpha = args.memory_alpha
        self.memory_beta0 = args.memory_beta0
        self.train_steps = float(train_steps)
        self.indices = range(self.memory_size)
        self.clear()

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.priority[self.index] = np.max(self.priority)
        self.index = (self.index + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self, batch_size):
        prob = self.priority / np.sum(self.priority)
        batch_idx = np.random.choice(self.indices, batch_size, False, prob)
        batch = [self.ring_buffer[i] for i in batch_idx]
        batch_prob = prob[batch_idx]
        return batch, batch_idx, batch_prob

    def get_batch_weights(self, batch_idx, batch_prob, iter_num):
        wt_end = min(iter_num / self.train_steps, 1.0)
        wt_start = 1.0 - wt_end
        beta_annealed = self.memory_beta0 * wt_start + 1.0 * wt_end
        batch_weights = (self.size * batch_prob)**(-beta_annealed)
        batch_weights /= np.max(batch_weights)
        return batch_weights

    def update_priority(self, batch_idx, batch_td_error):
        batch_priority = np.abs(batch_td_error)
        batch_priority[batch_priority > 1.0] = 1.0
        batch_priority[batch_priority == 0.0] = 1e-16
        self.priority[batch_idx] = batch_priority**self.memory_alpha

    def clear(self):
        self.ring_buffer = [None for _ in xrange(self.memory_size)]
        self.priority = np.zeros(self.memory_size)
        self.priority[0] = 1.0
        self.index = 0
        self.size = 0

    def print_status(self):
        print '  memory size: {:d}/{:d}'.format(self.size, self.memory_size)

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filepath):
        with open(filepath, 'rb') as save:
            memory = pickle.load(save)
        self.ring_buffer[:memory.size] = memory.ring_buffer[:memory.size]
        self.priority[:memory.size] = memory.priority[:memory.size]
        self.index = memory.size % self.memory_size
        self.size = memory.size

