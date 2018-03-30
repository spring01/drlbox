
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt


def padding(signal, window_size):
    pad = np.mean(signal[:window_size])
    return np.concatenate([[pad] * (window_size - 1), signal])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw a learning curve')
    parser.add_argument('--window-small', default=100, type=int,
        help='Small averaging window for instantaneous performance')
    parser.add_argument('--window-large', default=1000, type=int,
        help='Large averaging window for average performance')

    args, unknown_args = parser.parse_known_args()
    color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.5']
    if len(color_list) < len(unknown_args):
        for _ in range(len(unknown_args) - len(color_list)):
            color_list.append(np.random.rand(3))
    for name, color in zip(unknown_args, color_list):
        all_steps = []
        all_rewards = []
        step = 0
        with open(name) as out:
            for line in out:
                if 'episode reward' in line:
                    reward_list = re.findall('[+-]?\d+\.\d+', line)
                    if reward_list:
                        reward = float(reward_list[0])
                        all_steps.append(step)
                        all_rewards.append(reward)
                        step += 1
                if 'training step' in line:
                    step = int(re.findall('\d+', line)[0])

        num_episode = len(all_rewards)

        win_small = np.ones(args.window_small) / args.window_small
        padded_small = padding(all_rewards, args.window_small)
        mean_small = np.convolve(padded_small, win_small, mode='valid')

        win_large = np.ones(args.window_large) / args.window_large
        padded_large = padding(all_rewards, args.window_large)
        mean_large = np.convolve(padded_large, win_large, mode='valid')

        plt.plot(all_steps, mean_small, color=color, linewidth=4, alpha=0.1)
        plt.plot(all_steps, mean_large, color=color, linewidth=4, label=name)
    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.show()

