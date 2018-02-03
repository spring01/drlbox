
from .ac_eval import ACEvaluator
from .dqn_eval import DQNEvaluator
from .noisynet_ac_eval import NoisyNetACEvaluator
from .noisynet_dqn_eval import NoisyNetDQNEvaluator


EVALUATOR_CLS_DICT = {'ac':             ACEvaluator,
                      'noisynet-ac':    NoisyNetACEvaluator,
                      'dqn':            DQNEvaluator,
                      'noisynet-dqn':   NoisyNetDQNEvaluator,}

def make_evaluator(algorithm, **kwargs):
    algorithm = algorithm.lower()
    if algorithm not in EVALUATOR_CLS_DICT:
        raise ValueError('Algorithm "{}" not valid'.format(algorithm))
    return EVALUATOR_CLS_DICT[algorithm](**kwargs)

