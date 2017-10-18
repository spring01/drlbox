#!/bin/bash


output="$(mktemp -d)"
echo "made temp dir $output"

# cartpole a3c fc
python a3c_trainer.py \
    --rl_train_steps 500 --interval_save 200 --rl_save_path $output || exit 1

# cartpole evaluator
cartpole_weights="$(ls $output/CartPole-v0-run1/weights_*.p | tail -n 1)"
python evaluator.py --net_type acnet --read_weights $cartpole_weights \
    --policy_type stochastic --render false --eval_episodes 3 || exit 1

# breakout a3c fc
python a3c_trainer.py --dtf_num_workers 8 \
    --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --interval_save 200 --rl_save_path $output \
    --additional model.atari_nets fc 256 || exit 1

# breakout a3c lstm
python a3c_trainer.py --dtf_num_workers 8 \
    --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --interval_save 200 --rl_save_path $output \
    --additional model.atari_nets lstm 64 || exit 1

# breakout evaluator
breakout_weights="$(ls $output/Breakout-v0-run3/weights_*.p | tail -n 1)"
python evaluator.py --net_type acnet --read_weights $breakout_weights \
    --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --policy_type stochastic --render false --eval_episodes 3 \
    --additional model.atari_nets lstm 64 --net_type acnet || exit 1

echo "rm -rf $output"
rm -rf $output

