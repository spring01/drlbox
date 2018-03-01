
from .ac_eval import ACEvaluator
from .acer_eval import ACEREvaluator
from .dqn_eval import DQNEvaluator


EVALUATOR_CLS_DICT = {'ac':             ACEvaluator,
                      'acer':           ACEREvaluator,
                      'dqn':            DQNEvaluator,
                      }

def make_evaluator(algorithm, **kwargs):
    algorithm = algorithm.lower()
    if algorithm not in EVALUATOR_CLS_DICT:
        raise ValueError('Algorithm "{}" not valid'.format(algorithm))
    return EVALUATOR_CLS_DICT[algorithm](**kwargs)

