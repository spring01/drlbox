
import tensorflow as tf
import numpy as np
from drlbox.net.acer_net import ACERNet
from drlbox.common.util import discrete_action, softmax
from drlbox.common.policy import StochasticDisc, StochasticCont
from .trainer_base import Trainer


ACER_ACTION_SPACE_ONLY_DISC = 'action must be discrete in ACER network'

class ACERTrainer(Trainer):

    KEYWORD_DICT = {**Trainer.KEYWORD_DICT,
                    **dict(a3c_entropy_weight=1e-2,
                           acer_kl_weight=1e-1,)}
    net_cls = ACERNet
    minprob = 1e-4
    retrace_max = 1.0

    def setup_algorithm(self, action_space):
        self.loss_kwargs = dict(entropy_weight=self.a3c_entropy_weight,
                                kl_weight=self.acer_kl_weight)
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               clip_norm=self.opt_grad_clip_norm,
                               epsilon=self.opt_adam_epsilon)
        if discrete_action(action_space):
            self.policy = StochasticDisc()
        else:
            raise TypeError(ACER_ACTION_SPACE_ONLY_DISC)

    def setup_nets(self, worker_dev, rep_dev, env):
        super().setup_nets(worker_dev, rep_dev, env)
        with tf.device(rep_dev):
            self.average_net = self.build_net(env)
            self.average_net.set_sync_weights(self.global_net.weights)

    def set_session(self, sess):
        super().set_session(sess)
        self.average_net.set_session(sess)
        self.average_net.sync()

    def rollout_feed(self, rollout):
        r_state, r_input, r_action = rollout.state_input_action()

        # off-policy probabilities, length n
        r_act_logits = rollout.act_val()
        r_act_probs = self.softmax_with_minprob(r_act_logits)

        # on-policy probabilities and values, length n+1
        r_logits, r_q_val = self.online_net.ac_values(r_state)
        r_probs = self.softmax_with_minprob(r_logits)

        # likelihood ratio and retrace, length n
        r_lratio = r_probs[:-1] / r_act_probs
        r_retrace = np.minimum(self.retrace_max, r_lratio)

        # baseline, length n+1
        r_baseline = np.sum(r_probs * r_q_val, axis=1)
        #~ ForkedPdb().set_trace()

        # return, length n
        reward_long = 0.0 if rollout.done else r_baseline[-1]
        r_q_ret = np.zeros(len(rollout))
        for idx in reversed(range(len(rollout))):
            reward_long *= self.discount
            reward_long += rollout.reward_list[idx]
            r_q_ret[idx] = reward_long
            act = r_action[idx]
            val = r_q_val[idx, act]
            retrace = r_retrace[idx, act]
            reward_long = retrace * (reward_long - val) + r_baseline[idx]

        # logits from the average net, length n
        r_avg_logits = self.average_net.action_values(r_input)
        return (r_input, r_action, r_lratio, r_q_ret, r_q_val[:-1],
                r_baseline[:-1], r_avg_logits)

    def softmax_with_minprob(self, logits):
        return np.maximum(self.minprob, softmax(logits, axis=1))

import sys
import pdb

class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin
