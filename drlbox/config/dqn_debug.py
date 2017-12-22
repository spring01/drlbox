
'''
Reinforcement learning hyperparameters
'''
DISCOUNT        = 0.99
LEARNING_RATE   = 1e-4
TRAIN_STEPS     = 1000
BATCH_SIZE      = 32

'''
Intervals
'''
INTERVAL_SAVE           = 400
INTERVAL_TRAIN_ONLINE   = 4
INTERVAL_SYNC_TARGET    = 400

'''
Replay memory
'''
REPLAY_TYPE     = 'priority'
REPLAY_MAXLEN   = 100
REPLAY_MINLEN   = 50
REPLAY_ALPHA    = 0.5
REPLAY_BETA     = 0.4

'''
Policy
'''
POLICY_EPS_START    = 1.0
POLICY_EPS_END      = 0.1
POLICY_DECAY_STEPS  = 500

