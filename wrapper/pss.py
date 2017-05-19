import gym
import numpy as np
from PIL import Image


class PSSWrapper(gym.Wrapper):
    """
        A wrapper for frame Preprocessing, frame Stacking, and frame Skipping.
        Will convert input image to grayscale, resize to `resize`,
        stack `stack_frames` most recent resized frames together,
        and perform action for `act_steps` steps.
    """

    metadata = {'render.modes': ['human', 'grayscale_array']}

    def __init__(self, env, resize=(84, 110), stack_frames=4, act_steps=2):
        super(PSSWrapper, self).__init__(env)
        self.resize = resize
        self.stack_frames = stack_frames
        self.act_steps = act_steps
        self.stepcount = 0
        self.viewer = None

    def _step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.act_steps and not done:
            self.stepcount += 1
            obs, reward, done, info = self.env.step(action)
            self.state.pop(0)
            self.state.append(self.preprocess(obs))
            total_reward += reward
            current_step += 1
        if 'pss.stepcount' in info:
            raise gym.error.Error('Key "pss.stepcount" already in info. Make sure you are not stacking ' \
                                  'the PSSWrapper wrappers.')
        info['pss.stepcount'] = self.stepcount
        return self.state, total_reward, done, info

    def _reset(self):
        self.stepcount = 0
        obs = self.env.reset()
        self.state = [self.preprocess(obs)] * self.stack_frames
        return self.state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = np.stack(self.state, axis=2)
        if mode == 'grayscale_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = np.stack([np.max(img, axis=2)] * 3, axis=2)
            self.viewer.imshow(img)

    def preprocess(self, obs):
        img = Image.fromarray(obs)
        img = img.convert('L')
        img = img.resize(self.resize)
        return np.asarray(img)


