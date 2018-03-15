
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw learning curve')
    parser.add_argument('--window_small', default=100, type=int,
        help='Low level averaging window')
    parser.add_argument('--window_large', default=1000, type=int,
        help='High level averaging window')

    args, unknown_args = parser.parse_known_args()
    filename = unknown_args[0]
    all_rewards = []
    max_step = 0
    with open(filename) as out:
        for line in out:
            if 'episode reward' in line:
                reward_list = re.findall('[+-]?\d+\.\d+', line)
                if len(reward_list):
                    reward = float(reward_list[0])
                    all_rewards.append(reward)
            if 'training step' in line:
                max_step = int(re.findall('\d+', line)[0])

    num_episode = len(all_rewards)

    win_small = np.ones(args.window_small) / args.window_small
    mean_small = np.convolve(all_rewards, win_small, mode='valid')
    len_mean_small = len(mean_small)
    space_small = np.linspace(1, num_episode, len_mean_small)

    win_large = np.ones(args.window_large) / args.window_large
    mean_large = np.convolve(all_rewards, win_large, mode='valid')
    space_large = np.linspace(1, num_episode, len(mean_large))

    peak_pos = mean_small.argmax() / len_mean_small * max_step
    print('Peak position is at around step {:d}'.format(int(peak_pos)))

    plt.plot(space_small, mean_small, color='b', linewidth=4, alpha=0.2)
    plt.plot(space_large, mean_large, color='b', linewidth=4)
    plt.tight_layout(pad=0.1)
    plt.show()

