import gym
import numpy as np
from PIL import Image


class PPSSWrapper(gym.Wrapper):
    """
        A wrapper for frame PreProcessing, frame Stacking, and frame Skipping.
        Will convert input image to grayscale, resize to `resize`,
        stack `num_frames` most recent resized frames together,
        and perform action for `act_steps` steps.
    """

    metadata = {'render.modes': ['human', 'rgb_array']}

    '''
        Arguments for the constructor:
        env: Game environment to be PPSS wrapped;
        resize: Resized frame shape; tuple of 2 integers (width, height);
        num_frames: Number of frames to be stacked together; integer;
        act_steps: Number of actions performed in a row for skipping; integer.
    '''
    def __init__(self, env, resize=(84, 110), num_frames=4, act_steps=2):
        super(PPSSWrapper, self).__init__(env)
        width, height = resize
        shape = height, width, num_frames
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=shape)
        self.resize = resize
        self.num_frames = num_frames
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
        if 'ppss.stepcount' in info:
            raise gym.error.Error('Key "ppss.stepcount" already in info. '
                                  'Make sure you are not stacking '
                                  'the PPSSWrapper wrappers.')
        info['ppss.stepcount'] = self.stepcount
        self.stack_state = np.stack(self.state, axis=-1)
        return self.stack_state.astype(np.float32), total_reward, done, info

    def _reset(self):
        self.stepcount = 0
        obs = self.env.reset()
        self.state = [self.preprocess(obs)] * self.num_frames
        self.stack_state = np.stack(self.state, axis=-1)
        return self.stack_state.astype(np.float32)

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        if mode == 'rgb_array':
            return self.stack_state
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            img = np.stack([np.max(self.stack_state, axis=-1)] * 3, axis=-1)
            self.viewer.imshow(img)

    def preprocess(self, obs):
        img = Image.fromarray(obs)
        img = img.convert('L')
        img = img.resize(self.resize)
        return np.asarray(img)


