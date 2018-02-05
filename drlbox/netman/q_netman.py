
from .netman_base import NetManager

class QNetManager(NetManager):

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = rollout.state_input_action()
        last_state = r_state[-1:]
        online_last_value = self.online_net.action_values(last_state)[-1]
        target_last_value = self.target_net.action_values(last_state)[-1]
        target_last_q = target_last_value[np.argmax(online_last_value)]
        r_target = rollout.target(target_last_q)
        return r_input, r_action, r_target

    def sync_target(self):
        self.target_net.sync()
