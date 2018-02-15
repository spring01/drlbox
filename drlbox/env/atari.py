
import gym
from .envwrapper import HistoryStacker, RewardClipper, EpisodicLife


def make_env(name, num_frames=4, act_steps=2):
    env = gym.make(name)
    env = Preprocessor(env)
    env = HistoryStacker(env, num_frames, act_steps)
    env = RewardClipper(env, -1.0, 1.0)
    env = EpisodicLife(env)
    return env



'''
Atari game env preprocessor
'''
import numpy as np
from PIL import Image


class Preprocessor(gym.Wrapper):
    """
        A wrapper for frame preprocessing.
        Will convert input image to grayscale and resize to `resize`.
    """

    metadata = {'render.modes': ['human', 'wrapped', 'rgb_array']}
    resize = 84, 110 # tuple of 2 integers (height, width).

    '''
        Arguments for the constructor:
        env: Game environment to be preprocessed.
    '''
    def __init__(self, env):
        super().__init__(env)
        assert(isinstance(env.observation_space, gym.spaces.Box))
        assert(len(env.observation_space.shape) == 3)
        width, height = self.resize
        shape = height, width
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape,
                                                dtype=np.uint8)
        self.viewer = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.preprocessed_obs = self.preprocess(obs)
        return self.preprocessed_obs, reward, done, info

    def reset(self):
        self.preprocessed_obs = self.preprocess(self.env.reset())
        return self.preprocessed_obs

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return self.preprocessed_obs
        elif mode == 'human':
            self.env.render(mode='human')
        elif mode == 'wrapped':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = np.stack([self.preprocessed_obs] * 3, axis=2)
            self.viewer.imshow(img)

    def close(self):
        self.unwrapped.close()
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def preprocess(self, obs):
        img = Image.fromarray(obs)
        img = img.convert('L')
        img = img.resize(self.resize)
        return np.asarray(img)


