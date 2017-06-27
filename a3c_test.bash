#!/bin/bash


output="$(mktemp -d)"
echo "made temp dir $output"

# a3c fully connected
python a3c_atari_trainer.py --dtf_num_workers 8 --rl_train_steps 500 \
    --interval_save 200 --net_name 'fully connected' --net_size 256 \
    --rl_save_path $output

# a3c lstm
python a3c_atari_trainer.py --dtf_num_workers 8 --rl_train_steps 500 \
    --interval_save 200 --net_name 'lstm' --net_size 128 \
    --rl_save_path $output

# evaluator
trained_weights="$(ls $output/Breakout-v0-run2/weights_*.p | tail -n 1)"
python atari_evaluator.py --read_weights $trained_weights \
    --policy_type stochastic --render False \
    --net_type 'acnet' --net_name 'lstm' --net_size 128 \
    --eval_episodes 3

echo "rm -rf $output"
rm -rf $output

