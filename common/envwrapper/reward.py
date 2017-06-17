
import gym


class RewardClipper(gym.Wrapper):
    """
        A wrapper for reward clipping.
        Stacks `num_frames` most recent frames together,
        and performs action for `act_steps` steps each time env.step is called.
    """

    '''
        Arguments for the constructor:
        env: Game environment to be history stacking wrapped;
        num_frames: Number of frames to be stacked together; integer;
        act_steps: Number of actions performed between states; integer.
    '''
    def __init__(self, env, lower=-1.0, upper=1.0):
        super(RewardClipper, self).__init__(env)
        self.lower = lower
        self.upper = upper

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = min(max(reward, self.lower), self.upper)
        return obs, reward, done, info


