#!/bin/bash


output="$(mktemp -d)"
echo "made temp dir $output"

# a3c fully connected
python a3c_atari_trainer.py --dtf_num_workers 8 --rl_train_steps 500 --interval_save 200 \
    --net_name 'fully connected' --net_size 256 --rl_save_path $output

# a3c gru
python a3c_atari_trainer.py --dtf_num_workers 8 --rl_train_steps 500 --interval_save 200 \
    --net_name 'gru' --net_size 256 --rl_save_path $output

# evaluator
python atari_evaluator.py --read_weights $output/Breakout-v0-run2/weights_4*.p \
    --policy_type stochastic --render False \
    --net_type 'acnet' --net_name 'gru' --net_size 256 \
    --eval_episodes 3

echo "rm -rf $output"
rm -rf $output

