#!/bin/bash


output="$(mktemp -d)"
echo "made temp dir $output"

# dqn
python dqn_atari_trainer.py --dqn_train_steps 500 \
    --dqn_sync_target_interval 200 --dqn_save_interval 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --qnet_name 'dqn' --qnet_size 256 --dqn_output $output

# dueling dqn
python dqn_atari_trainer.py --dqn_train_steps 500 \
    --dqn_sync_target_interval 200 --dqn_save_interval 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --qnet_name 'dueling dqn' --qnet_size 256 --dqn_output $output

# drqn lstm
python dqn_atari_trainer.py --dqn_train_steps 500 \
    --dqn_sync_target_interval 200 --dqn_save_interval 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --qnet_name 'drqn gru' --qnet_size 128 --dqn_output $output

# dueling drqn gru
python dqn_atari_trainer.py --dqn_train_steps 500 \
    --dqn_sync_target_interval 200 --dqn_save_interval 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --qnet_name 'dueling drqn gru' --qnet_size 128 --dqn_output $output

# evaluator
python atari_evaluator.py --read_weights $output/Breakout-v0-run2/weights_4*.p \
    --policy_eps 0.05 --render False \
    --net_type 'qnet' --net_name 'dueling dqn' --net_size 256 \
    --eval_episodes 3

echo "rm -rf $output"
rm -rf $output

