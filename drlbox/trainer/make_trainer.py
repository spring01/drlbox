
from .a3c_trainer import A3CTrainer
from .acktr_trainer import ACKTRTrainer
from .dqn_trainer import DQNTrainer
from .noisynet_a3c_trainer import NoisyNetA3CTrainer
from .noisynet_dqn_trainer import NoisyNetDQNTrainer


TRAINER_CLS_DICT = {'a3c':          A3CTrainer,
                    'acktr':        ACKTRTrainer,
                    'noisynet-a3c': NoisyNetA3CTrainer,
                    'dqn':          DQNTrainer,
                    'noisynet-dqn': NoisyNetDQNTrainer,}

def make_trainer(algorithm, **kwargs):
    algorithm = algorithm.lower()
    if algorithm not in TRAINER_CLS_DICT:
        raise ValueError('Algorithm "{}" not valid'.format(algorithm))
    return TRAINER_CLS_DICT[algorithm](**kwargs)

