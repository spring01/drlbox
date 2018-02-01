
from .rollout_base import Rollout


class RolloutAC(Rollout):

    def get_feed(self, target_net, online_net):
        rollout_state, rollout_input, rollout_action = self.state_input_action()
        rollout_value = target_net.state_value(rollout_state)
        rollout_target = self.target(rollout_value[-1])
        rollout_adv = rollout_target - rollout_value[:-1]
        return rollout_input, rollout_action, rollout_adv, rollout_target

