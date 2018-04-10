
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict


def padding(signal, window_size):
    pad = np.mean(signal[:window_size])
    return np.concatenate([[pad] * (window_size - 1), signal])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draw a learning curve')
    parser.add_argument('--window-small', default=100, type=int,
        help='Small averaging window for instantaneous performance')
    parser.add_argument('--window-large', default=1000, type=int,
        help='Large averaging window for average performance')
    parser.add_argument('--linewidth', default=3.0, type=float,
        help='Line width')
    parser.add_argument('--colors', default=None, type=str, nargs='+',
        help='Color list')
    parser.add_argument('--labels', default=None, type=str, nargs='+',
        help='Labels for making the legend')

    args, unknown_args = parser.parse_known_args()

    # colors
    if args.colors is None:
        color_list = ['b', 'g', 'r', 'c', 'm', 'y', 'k', '0.5']
        if len(color_list) < len(unknown_args):
            for _ in range(len(unknown_args) - len(color_list)):
                color_list.append(np.random.rand(3))
    else:
        color_list = []
        for color, number in zip(args.colors[:-1:2], args.colors[1::2]):
            color_list.extend([color] * int(number))

    # labels
    if args.labels is None:
        label_list = unknown_args
    else:
        label_list = []
        for label, number in zip(args.labels[:-1:2], args.labels[1::2]):
            label_list.extend([label] * int(number))

    # plot a learning curve for each file
    for filename, color, label in zip(unknown_args, color_list, label_list):
        # find rewards and steps
        step_reward = OrderedDict()
        step = 0
        with open(filename) as out:
            for line in out:
                if 'episode reward' in line:
                    reward_list = re.findall('[+-]?\d+\.\d+', line)
                    if reward_list:
                        reward = float(reward_list[0])
                        if step in step_reward:
                            step_reward[step].append(reward)
                        else:
                            step_reward[step] = [reward]
                if 'training step' in line:
                    step_next = int(re.findall('\d+', line)[0])
                    if step_next > step:
                        step = step_next

        # interpolate steps to arrange episodic rewards
        step_list = list(step_reward.keys())
        int_step_reward = OrderedDict()
        for step, next_step in zip(step_list[:-1], step_list[1:]):
            int_step_array = np.linspace(step, next_step,
                                         len(step_reward[step]),
                                         endpoint=False)
            for int_step, reward in zip(int_step_array, step_reward[step]):
                int_step_reward[int_step] = reward
        all_steps = list(int_step_reward.keys())
        all_rewards = [int_step_reward[step] for step in all_steps]

        if args.window_small:
            win_small = np.ones(args.window_small) / args.window_small
            padded_small = padding(all_rewards, args.window_small)
            mean_small = np.convolve(padded_small, win_small, mode='valid')
            plt.plot(all_steps, mean_small, color=color,
                     linewidth=args.linewidth, alpha=0.1)

        if args.window_large:
            win_large = np.ones(args.window_large) / args.window_large
            padded_large = padding(all_rewards, args.window_large)
            mean_large = np.convolve(padded_large, win_large, mode='valid')
            plt.plot(all_steps, mean_large, color=color,
                     linewidth=args.linewidth, label=label)

    plt.tight_layout(pad=0.1)
    plt.legend()
    plt.show()

