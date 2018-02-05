

class NetManager:

    def __init__(self):
        pass

    def sync_online(self):
        self.online_net.sync()

    def action_values(self, state):
        return self.online_net.action_values([state])[0]

    def train_on_rollout_list(self, rollout_list):
        train_args = self.train_args(rollout_list)
        batch_loss = self.online_net.train_on_batch(*train_args)
        return batch_loss

    def train_args(self, rollout_list)
        feed_list = [self.rollout_feed(rollout) for rollout in rollout_list]
        return list(map(concatenate, zip(*feed_list)))

    def rollout_feed(self, rollout):
        raise NotImplementedError

    def sync_target(self):
        pass

