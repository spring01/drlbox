
import multiprocessing

NUM_WORKERS     = multiprocessing.cpu_count()
PORT_BEGIN      = 2220
DISCOUNT        = 0.99
LEARNING_RATE   = 1e-4
ADAM_EPSILON    = 1e-4
GRAD_CLIP_NORM  = 40.0
ENTROPY_WEIGHT  = 0.01
ROLLOUT_MAXLEN  = 20
BATCH_SIZE      = 32
TRAIN_STEPS     = 1000000
INTERVAL_SAVE   = 10000

'''
Policy
'''
POLICY_EPS_START    = 1.0
POLICY_EPS_END      = 0.1
POLICY_DECAY_STEPS  = 1000000

'''
Continuous action policy
'''
CONT_POLICY_MIN_VAR = 1E-8

'''
K-fac
'''
KFAC_COV_EMA_DECAY      = 0.95
KFAC_DAMPING            = 1E-3
KFAC_TRUST_RADIUS       = 2E-3
KFAC_INV_UPD_INTERVAL   = 100

