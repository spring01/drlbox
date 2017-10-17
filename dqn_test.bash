#!/bin/bash


output="$(mktemp -d)"
echo "made temp dir $output"

# fully connected
python dqn_atari_trainer.py --rl_train_steps 500 \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --net_name 'fc' --net_size 256 --rl_save_path $output

# dueling fully connected
python dqn_atari_trainer.py --rl_train_steps 500 \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --net_name 'dueling fc' --net_size 256 --rl_save_path $output

# lstm
python dqn_atari_trainer.py --rl_train_steps 500 \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --net_name 'lstm' --net_size 128 --rl_save_path $output

# dueling gru
python dqn_atari_trainer.py --rl_train_steps 500 \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --net_name 'dueling gru' --net_size 128 --rl_save_path $output

# evaluator
python atari_evaluator.py --read_weights $output/Breakout-v0-run2/weights_4*.p \
    --policy_eps 0.05 --render False \
    --net_type 'qnet' --net_name 'dueling fc' --net_size 256 \
    --eval_episodes 3

echo "rm -rf $output"
rm -rf $output

