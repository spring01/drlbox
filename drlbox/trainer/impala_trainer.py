
import numpy as np
from drlbox.common.util import softmax_with_minprob
from drlbox.common.replay import Replay
from .a3c_trainer import A3CTrainer


class IMPALATrainer(A3CTrainer):

    KEYWORD_DICT = {**A3CTrainer.KEYWORD_DICT,
                    **dict(impala_trunc_rho_max=1.0,
                           impala_trunc_c_max=1.0,
                           replay_maxlen=1000,
                           replay_minlen=100,
                           replay_ratio=4,)}
    softmax_minprob = 1e-6

    def setup_nets(self, worker_dev, rep_dev, env):
        super().setup_nets(worker_dev, rep_dev, env)
        self.replay = Replay(self.replay_maxlen, self.replay_minlen)

    def train_on_rollout_list(self, rollout_list):
        batch_loss = super().train_on_rollout_list(rollout_list)
        loss_list = [batch_loss]
        self.replay.append(rollout_list)
        if len(self.replay) >= self.replay_minlen:
            replay_times = np.random.poisson(self.replay_ratio)
            rep_list, rep_idx, rep_weight = self.replay.sample(replay_times)
            for roll_list, idx, weight in zip(rep_list, rep_idx, rep_weight):
                self.online_net.sync()
                batch_loss = super().train_on_rollout_list(roll_list)
                loss_list.append(batch_loss)
        return np.mean(loss_list)

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = self.rollout_state_input_action(rollout)

        # off-policy probabilities, length n
        r_act_logits = np.stack(rollout.act_val_list)
        r_act_probs = softmax_with_minprob(r_act_logits, self.softmax_minprob)
        r_act_probs_act = r_act_probs[(range(len(r_act_probs)), r_action)]

        # on-policy probabilities, length n
        r_logits = self.online_net.action_values(r_input)
        r_probs = softmax_with_minprob(r_logits, self.softmax_minprob)
        r_probs_act = r_probs[(range(len(r_probs)), r_action)]

        # on-policy values, length n+1
        r_value = self.online_net.state_value(r_state)

        # likelihood ratio, length n
        r_lratio_act = r_probs_act / r_act_probs_act
        r_trunc_rho = np.minimum(self.impala_trunc_rho_max, r_lratio_act)
        r_trunc_c = np.minimum(self.impala_trunc_c_max, r_lratio_act)

        # v-trace target, length n+1
        r_reward = np.stack(rollout.reward_list)
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
        return r_input, r_action, r_adv, r_target[:-1]

