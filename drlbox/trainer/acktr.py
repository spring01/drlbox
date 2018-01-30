
from .a3c import A3CTrainer
from drlbox.net import ACKTRNet
from drlbox.async.rollout import RolloutAC
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import StochasticDisc, StochasticCont


class ACKTRTrainer(A3CTrainer):

    KEYWORD_DICT = {**A3CTrainer.KEYWORD_DICT,
                    **dict(kfac_cov_ema_decay=0.95,
                           kfac_damping=1e-3,
                           kfac_trust_radius=1e-3,
                           kfac_inv_upd_interval=10,)}

    def setup_algorithm(self, action_space):
        super.setup_algorithm(action_space)
        self.net_cls = ACKTRNet
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               cov_ema_decay=self.kfac_cov_ema_decay,
                               damping=self.kfac_damping,
                               trust_radius=self.kfac_trust_radius,
                               inv_upd_interval=self.kfac_inv_upd_interval)

