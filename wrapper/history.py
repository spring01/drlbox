import gym
import numpy as np


class HistoryWrapper(gym.Wrapper):
    """
        A wrapper for history stacking.
        Stacks `num_frames` most recent frames together,
        and performs action for `act_steps` steps each time env.step is called.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    '''
        Arguments for the constructor:
        env: Game environment to be history stacking wrapped;
        num_frames: Number of frames to be stacked together; integer;
        act_steps: Number of actions performed between states; integer.
    '''
    def __init__(self, env, num_frames=4, act_steps=2):
        super(HistoryWrapper, self).__init__(env)
        height, width = self.env.observation_space.shape
        shape = height, width, num_frames
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)
        self.num_frames = num_frames
        self.act_steps = act_steps
        self.viewer = None

    def _step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.act_steps and not done:
            obs, reward, done, info = self.env.step(action)
            self.obs_list.pop(0)
            self.obs_list.append(obs)
            total_reward += reward
            current_step += 1
        self.state = np.stack(self.obs_list, axis=2)
        return self.state.astype(np.float32), total_reward, done, info

    def _reset(self):
        obs = self.env.reset()
        self.obs_list = [obs] * self.num_frames
        self.state = np.stack(self.obs_list, axis=2)
        return self.state.astype(np.float32)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'grayscale_array':
            return self.state
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = np.stack([np.max(self.state, axis=2)] * 3, axis=2)
            self.viewer.imshow(img)


