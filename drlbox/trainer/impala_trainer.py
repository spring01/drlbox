
import numpy as np
from drlbox.common.util import softmax_with_minprob
from .a3c_trainer import A3CTrainer


IMPALA_KWARGS = dict(
    impala_trunc_rho_max=1.0,
    impala_trunc_c_max=1.0,
    replay_type='uniform',
    )

class IMPALATrainer(A3CTrainer):

    KWARGS = {**A3CTrainer.KWARGS, **IMPALA_KWARGS}
    softmax_minprob = 1e-6

    def concat_bootstrap(self, cc_state, rl_slice):
        cc_logits, cc_value = self.online_net.ac_values(cc_state)
        return cc_logits, cc_value

    def rollout_feed(self, rollout, r_logits, r_value):
        r_action = np.array(rollout.action_list)

        # off-policy probabilities, length n
        r_off_logits = np.array(rollout.act_val_list)
        r_off_probs = softmax_with_minprob(r_off_logits, self.softmax_minprob)
        r_off_probs_act = r_off_probs[(range(len(r_off_probs)), r_action)]

        # on-policy probabilities, length n
        r_logits = r_logits[:-1]
        r_probs = softmax_with_minprob(r_logits, self.softmax_minprob)
        r_probs_act = r_probs[(range(len(r_probs)), r_action)]

        # likelihood ratio, length n
        r_lratio_act = r_probs_act / r_off_probs_act
        r_trunc_rho = np.minimum(self.impala_trunc_rho_max, r_lratio_act)
        r_trunc_c = np.minimum(self.impala_trunc_c_max, r_lratio_act)

        # v-trace target, length n+1
        r_reward = np.array(rollout.reward_list)
        if rollout.done:
            r_value[-1] = 0.0
        r_target = np.zeros(len(rollout) + 1)
        r_target[-1] = r_value[-1]
        for idx in reversed(range(len(rollout))):
            trunc_rho = r_trunc_rho[idx]
            trunc_c = r_trunc_c[idx]
            value = r_value[idx]
            reward = r_reward[idx]
            value_next = r_value[idx + 1]
            target_next = r_target[idx + 1]
            dv_rho = trunc_rho * (reward + self.discount * value_next - value)
            dv_c = trunc_c * self.discount * (target_next - value_next)
            r_target[idx] = value + dv_rho + dv_c

        # advantage
        r_adv = r_reward + self.discount * r_target[1:] - r_value[:-1]
        return r_action, r_adv, r_target[:-1]

