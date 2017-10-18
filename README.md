# HCDRL: HC's playground of deep reinforcement learning
Supports *only Python3* (oops).

## Requirements:
- tensorflow>=1.3
- gym[atari]

## Minimal sample usage (of CartPole):
- DQN training: `python dqn_trainer.py`
- A3C training: `python a3c_trainer.py`
- Evaluation: `python evaluator.py --read_weights SAVED_WEIGHTS`

## (Relatively) Full tests, including CartPole and Atari Breakout:
- DQN: `bash dqn_test.bash`
- A3C: `bash a3c_test.bash`

