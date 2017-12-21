
import multiprocessing

NUM_WORKERS = multiprocessing.cpu_count()
PORT_BEGIN  = 2220

'''
Reinforcement learning hyperparameters
'''
RL_DISCOUNT         = 0.99
RL_LEARNING_RATE    = 1e-4
RL_TRAIN_STEPS      = 1000000
RL_ENTROPY_WEIGHT   = 0.01
RL_INTERVAL_SAVE    = 10000
RL_ROLLOUT_MAXLEN   = 20

