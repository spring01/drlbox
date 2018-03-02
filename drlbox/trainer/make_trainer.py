
from .a3c_trainer import A3CTrainer
from .acer_trainer import ACERTrainer
from .impala_trainer import IMPALATrainer
from .dqn_trainer import DQNTrainer


TRAINER_CLS_DICT = {'a3c':              A3CTrainer,
                    'acer':             ACERTrainer,
                    'impala':           IMPALATrainer,
                    'dqn':              DQNTrainer,}

def make_trainer(algorithm, **kwargs):
    algorithm = algorithm.lower()
    if algorithm not in TRAINER_CLS_DICT:
        raise ValueError('Algorithm "{}" not valid'.format(algorithm))
    return TRAINER_CLS_DICT[algorithm](**kwargs)

