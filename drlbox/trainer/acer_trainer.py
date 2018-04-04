
import numpy as np
import tensorflow as tf
from drlbox.net import ACERNet
from drlbox.common.util import softmax_with_minprob
from drlbox.common.policy import SoftmaxPolicy
from .a3c_trainer import A3CTrainer


ACER_KWARGS = dict(
    acer_trunc_max=10.0,
    acer_kl_weight=0.0,
    acer_soft_update_ratio=0.05,
    replay_type='uniform',
    )

class ACERTrainer(A3CTrainer):

    KWARGS = {**A3CTrainer.KWARGS, **ACER_KWARGS}
    net_cls = ACERNet
    softmax_minprob = 1e-6
    retrace_max = 1.0

    def setup_algorithm(self):
        super().setup_algorithm()
        assert self.action_mode == 'discrete'
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                kl_weight=self.acer_kl_weight,
                                trunc_max=self.acer_trunc_max,
                                policy_type='softmax')
        self.model_kwargs['size_value'] = self.action_dim
        self.policy = SoftmaxPolicy()

    def setup_nets(self, worker_dev, rep_dev, env):
        super().setup_nets(worker_dev, rep_dev, env)
        if self.acer_kl_weight:
            with tf.device(rep_dev):
                self.average_net = self.build_net(env)
                self.average_net.set_sync_weights(self.global_net.weights)
                self.average_net.set_soft_update(self.global_net.weights,
                                                 self.acer_soft_update_ratio)

    def set_session(self, sess):
        super().set_session(sess)
        if self.acer_kl_weight:
            self.average_net.set_session(sess)
            self.average_net.sync()

    def sync_to_global(self):
        super().sync_to_global()
        if self.acer_kl_weight and self.noisynet is not None:
            self.average_net.sample_noise()

    def train_on_batch(self, *args):
        batch_result = super().train_on_batch(*args)
        if self.acer_kl_weight:
            self.average_net.soft_update()
        return batch_result

    def concat_bootstrap(self, cc_state, rl_slice):
        cc_logits, cc_boot = self.online_net.ac_values(cc_state)
        if self.acer_kl_weight:
            cc_avg_logits = self.average_net.action_values(cc_state)
            return cc_logits, cc_boot, cc_avg_logits
        else:
            return cc_logits, cc_boot

    def rollout_feed(self, *args):
        if self.acer_kl_weight:
            rollout, r_logits, r_boot, r_avg_logits = args
        else:
            rollout, r_logits, r_boot = args
        r_action = np.array(rollout.action_list)

        # off-policy probabilities, length n
        r_off_logits = np.array(rollout.act_val_list)
        r_off_probs = softmax_with_minprob(r_off_logits, self.softmax_minprob)

        # on-policy probabilities and values, length n+1
        r_probs = softmax_with_minprob(r_logits, self.softmax_minprob)

        # likelihood ratio and retrace, length n
        r_lratio = r_probs[:-1] / r_off_probs
        r_retrace = np.minimum(self.retrace_max, r_lratio)

        # baseline, length n+1
        r_baseline = np.sum(r_probs * r_boot, axis=1)

        # return, length n
        reward_long = 0.0 if rollout.done else r_baseline[-1]
        r_target = np.zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_target[idx] = reward_long
            act = r_action[idx]
            val = r_boot[idx, act]
            retrace = r_retrace[idx, act]
            reward_long = retrace * (reward_long - val) + r_baseline[idx]

        # logits from the average net, length n
        result = r_action, r_lratio, r_target, r_boot[:-1], r_baseline[:-1]
        if self.acer_kl_weight:
            return (*result, r_avg_logits[:-1])
        else:
            return result

