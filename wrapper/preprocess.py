import gym
import numpy as np
from PIL import Image


class PreprocessWrapper(gym.Wrapper):
    """
        A wrapper for frame preprocessing.
        Will convert input image to grayscale and resize to `resize`.
    """

    metadata = {'render.modes': ['human', 'wrapped', 'rgb_array']}

    '''
        Arguments for the constructor:
        env: Game environment to be preprocessed;
        resize: Resized frame shape; tuple of 2 integers (height, width).
    '''
    def __init__(self, env, resize=(84, 110)):
        super(PreprocessWrapper, self).__init__(env)
        assert(isinstance(env.observation_space, gym.spaces.Box))
        assert(len(env.observation_space.shape) == 3)
        width, height = resize
        shape = height, width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)
        self.resize = resize
        self.viewer = None

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.preprocessed_obs = self.preprocess(obs)
        return self.preprocessed_obs, reward, done, info

    def _reset(self):
        self.preprocessed_obs = self.preprocess(self.env.reset())
        return self.preprocessed_obs

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self.preprocessed_obs
        elif mode == 'human':
            self.env.render()
        elif mode == 'wrapped':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = np.stack([self.preprocessed_obs] * 3, axis=2)
            self.viewer.imshow(img)

    def preprocess(self, obs):
        img = Image.fromarray(obs)
        img = img.convert('L')
        img = img.resize(self.resize)
        return np.asarray(img)


