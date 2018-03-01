
from .a3c_trainer import A3CTrainer
from .acktr_trainer import ACKTRTrainer
from .acer_trainer import ACERTrainer
from .acerktr_trainer import ACERKTRTrainer
from .impala_trainer import IMPALATrainer
from .dqn_trainer import DQNTrainer


TRAINER_CLS_DICT = {'a3c':              A3CTrainer,
                    'acktr':            ACKTRTrainer,
                    'acer':             ACERTrainer,
                    'acerktr':          ACERKTRTrainer,
                    'impala':           IMPALATrainer,
                    'dqn':              DQNTrainer,}

def make_trainer(algorithm, **kwargs):
    algorithm = algorithm.lower()
    if algorithm not in TRAINER_CLS_DICT:
        raise ValueError('Algorithm "{}" not valid'.format(algorithm))
    return TRAINER_CLS_DICT[algorithm](**kwargs)

