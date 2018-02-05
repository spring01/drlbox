
from .netman_base import NetManager

class ACNetManager(NetManager):

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = rollout.state_input_action()
        r_value = self.online_net.state_value(r_state)
        r_target = rollout.target(r_value[-1])
        r_adv = r_target - r_value[:-1]
        return r_input, r_action, r_adv, r_target

