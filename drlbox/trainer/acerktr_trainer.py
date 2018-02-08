
from drlbox.net import ACERKTRNet
from .acer_trainer import ACERTrainer
from .acktr_trainer import ACKTRTrainer


class ACERKTRTrainer(ACERTrainer, ACKTRTrainer):

    KEYWORD_DICT = {**ACERTrainer.KEYWORD_DICT,
                    **ACKTRTrainer.KEYWORD_DICT}
    net_cls = ACERKTRNet

    def setup_algorithm(self, action_space):
        super().setup_algorithm(action_space)
        self.opt_kwargs = dict(learning_rate=self.opt_learning_rate,
                               cov_ema_decay=self.kfac_cov_ema_decay,
                               damping=self.kfac_damping,
                               trust_radius=self.kfac_trust_radius,
                               inv_upd_interval=self.kfac_inv_upd_interval)

