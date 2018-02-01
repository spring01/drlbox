
import numpy as np
from .rollout_base import Rollout


class RolloutMultiStepQ(Rollout):

    def get_feed(self, target_net, online_net):
        rollout_state, rollout_input, rollout_action = self.state_input_action()
        last_state = rollout_state[-1:]
        online_last_value = online_net.action_values(last_state)[-1]
        target_last_value = target_net.action_values(last_state)[-1]
        target_last_q = target_last_value[np.argmax(online_last_value)]
        rollout_target = self.target(target_last_q)
        return rollout_input, rollout_action, rollout_target

