#!/bin/bash

function clean_exit() {
    echo "rm -rf $output"
    rm -rf $output
    exit $1
}

output="$(mktemp -d)"
echo "made temp dir $output"

# cartpole dqn
python dqn_trainer.py --rl_train_steps 500 --rl_save_path $output \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 || clean_exit 1

# cartpole evaluator
cartpole_weights="$(ls $output/CartPole-v0-run1/weights_*.p | tail -n 1)"
python evaluator.py --net_type qnet --read_weights $cartpole_weights \
    --policy_eps 0.05 --render false --eval_episodes 3 || clean_exit 1

# breakout dqn fc
python dqn_trainer.py --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --rl_save_path $output \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --additional hcdrl.model.atari_nets 'fc' 256 || clean_exit 1

# breakout dqn dueling fc
python dqn_trainer.py --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --rl_save_path $output \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --additional hcdrl.model.atari_nets 'dueling fc' 256 || clean_exit 1

# breakout dqn lstm
python dqn_trainer.py --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --rl_save_path $output \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --additional hcdrl.model.atari_nets 'lstm' 64 || clean_exit 1

# breakout dqn dueling gru
python dqn_trainer.py --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --rl_train_steps 500 --rl_save_path $output \
    --interval_sync_target 200 --interval_save 200 \
    --memory_maxlen 200 --memory_fill 100 \
    --additional hcdrl.model.atari_nets 'dueling gru' 64 || clean_exit 1

# breakout evaluator
breakout_weights="$(ls $output/Breakout-v0-run3/weights_*.p | tail -n 1)"
python evaluator.py --env Breakout-v0 --env_num_frames 4 --env_act_steps 4 \
    --net_type qnet --read_weights $breakout_weights \
    --policy_eps 0.05 --render false --eval_episodes 3 \
    --additional hcdrl.model.atari_nets 'dueling fc' 256 || clean_exit 1

clean_exit 0

