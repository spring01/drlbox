
import random
import pickle


FILL_PERCENT = 0.1

'''
Ring-buffer uniformly sampled replay memory.
Both 'append' and 'sample' are O(1)
'''
class Replay:

    def __init__(self, maxlen, minlen=None):
        self.maxlen = maxlen
        if minlen is None:
            self.minlen = FILL_PERCENT * self.maxlen
        else:
            self.minlen = minlen
        self.ring_buffer = [None] * self.maxlen
        self.index = 0
        self.length = 0

    def append(self, transition):
        self.ring_buffer[self.index] = transition
        self.index = (self.index + 1) % self.maxlen
        self.length = min(self.length + 1, self.maxlen)

    def extend(self, batch):
        for transition in batch:
            self.append(transition)

    def sample(self, batch_size):
        batch_idx = [random.randrange(len(self)) for _ in range(batch_size)]
        batch = [self.ring_buffer[i] for i in batch_idx]
        return batch, batch_idx, [None] * batch_size

    def __len__(self):
        return self.length

    def usable(self):
        return len(self) >= self.minlen

    def save(self, filepath):
        with open(filepath, 'wb') as save:
            pickle.dump(self, save, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as save:
            replay = pickle.load(save)
        return replay


'''
Proportional prioritization with ring-buffer and sum-tree indexing.
'''
class PriorityReplay(Replay):

    error_eps = 1e-2

    def __init__(self, maxlen, minlen=None,
                 alpha=0.6, beta=0.4, beta_delta=1e-8):
        super().__init__(maxlen, minlen)
        self.sum_tree = [0.0] * (2 * self.maxlen - 1)
        self.sum_tree_kickout = [0.0] * (2 * self.maxlen - 1)
        self.alpha = alpha
        self.beta = beta
        self.beta_delta = beta_delta
        self.max_priority = 1.0         # max priority we've ever seen so far

    def append(self, transition):
        self.update_sumtree(self.index, self.max_priority)
        super().append(transition)

    def sample(self, batch_size):
        len_seg = self.sum_tree[0] / batch_size
        rand_list = [(random.random() + i) * len_seg for i in range(batch_size)]
        batch, batch_idx, batch_weights = [], [], []
        for rand in rand_list:
            ring_idx, priority, transition = self.get_leaf(rand)
            batch.append(transition)
            batch_idx.append(ring_idx)
            weight = (priority / self.sum_tree[0])**(-self.beta)
            batch_weights.append(weight)
        max_weight = max(batch_weights)
        batch_weights = [w / max_weight for w in batch_weights]
        self.beta = min(1.0, self.beta + self.beta_delta)
        return batch, batch_idx, batch_weights

    def update_priority(self, batch_idx, batch_error):
        for ring_idx, error in zip(batch_idx, batch_error):
            priority = abs(error) + self.error_eps
            priority **= self.alpha
            self.max_priority = max(self.max_priority, priority)
            self.update_sumtree(ring_idx, priority)

    def get_leaf(self, value):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions
        Array type for storing:
        [0, 1, 2, 3, 4, 5, 6]
        """
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.sum_tree):  # reach bottom, end search
                leaf_idx = parent
                break
            else:
                if value <= self.sum_tree[left]:
                    parent = left
                else:
                    value -= self.sum_tree[left]
                    parent = right
        ring_idx = leaf_idx - (self.maxlen - 1)
        return ring_idx, self.sum_tree[leaf_idx], self.ring_buffer[ring_idx]

    def update_sumtree(self, ring_idx, priority):
        leaf_idx = ring_idx + (self.maxlen - 1)
        change = priority - self.sum_tree[leaf_idx]
        self.sum_tree[leaf_idx] = priority
        while leaf_idx:
            leaf_idx = (leaf_idx - 1) // 2
            self.sum_tree[leaf_idx] += change


