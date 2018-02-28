
import tensorflow as tf
import numpy as np
from drlbox.net import ACERNet
from drlbox.common.util import discrete_action, softmax_with_minprob
from drlbox.common.policy import SoftmaxPolicy
from drlbox.common.replay import Replay
from .a3c_trainer import A3CTrainer


ACER_ACTION_SPACE_ONLY_DISC = 'action_space must be discrete in ACER network'

class ACERTrainer(A3CTrainer):

    KEYWORD_DICT = {**A3CTrainer.KEYWORD_DICT,
                    **dict(acer_kl_weight=1e-1,
                           acer_trunc_max=10.0,
                           acer_soft_update_ratio=0.05,
                           replay_maxlen=1000,
                           replay_minlen=100,
                           replay_ratio=4,)}
    net_cls = ACERNet
    minprob = 1e-6
    retrace_max = 1.0

    def setup_algorithm(self, action_space):
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                kl_weight=self.acer_kl_weight,
                                trunc_max=self.acer_trunc_max)
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)
        if discrete_action(action_space):
            self.policy = SoftmaxPolicy()
        else:
            raise TypeError(ACER_ACTION_SPACE_ONLY_DISC)

    def setup_nets(self, worker_dev, rep_dev, env):
        super().setup_nets(worker_dev, rep_dev, env)
        with tf.device(rep_dev):
            self.average_net = self.build_net(env)
            self.average_net.set_sync_weights(self.global_net.weights)
            self.average_net.set_soft_update(self.global_net.weights,
                                             self.acer_soft_update_ratio)
        self.replay = Replay(self.replay_maxlen, self.replay_minlen)

    def set_session(self, sess):
        super().set_session(sess)
        self.average_net.set_session(sess)
        self.average_net.sync()

    def train_on_rollout_list(self, rollout_list):
        batch_loss = super().train_on_rollout_list(rollout_list)
        self.average_net.soft_update()
        loss_list = [batch_loss]
        self.replay.append(rollout_list)
        if len(self.replay) >= self.replay_minlen:
            replay_times = np.random.poisson(self.replay_ratio)
            rep_list, rep_idx, rep_weight = self.replay.sample(replay_times)
            for roll_list, idx, weight in zip(rep_list, rep_idx, rep_weight):
                self.online_net.sync()
                batch_loss = super().train_on_rollout_list(roll_list)
                self.average_net.soft_update()
                loss_list.append(batch_loss)
        return np.mean(loss_list)

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = self.rollout_state_input_action(rollout)

        # off-policy probabilities, length n
        r_act_logits = np.stack(rollout.act_val_list)
        r_act_probs = softmax_with_minprob(r_act_logits, self.minprob)

        # on-policy probabilities and values, length n+1
        r_logits, r_boot_value = self.online_net.ac_values(r_state)
        r_probs = softmax_with_minprob(r_logits, self.minprob)

        # likelihood ratio and retrace, length n
        r_lratio = r_probs[:-1] / r_act_probs
        r_retrace = np.minimum(self.retrace_max, r_lratio)

        # baseline, length n+1
        r_baseline = np.sum(r_probs * r_boot_value, axis=1)

        # return, length n
        reward_long = 0.0 if rollout.done else r_baseline[-1]
        r_sample_return = np.zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_sample_return[idx] = reward_long
            act = r_action[idx]
            val = r_boot_value[idx, act]
            retrace = r_retrace[idx, act]
            reward_long = retrace * (reward_long - val) + r_baseline[idx]

        # logits from the average net, length n
        r_avg_logits = self.average_net.action_values(r_input)
        return (r_input, r_action, r_lratio, r_sample_return, r_boot_value[:-1],
                r_baseline[:-1], r_avg_logits)


