
import gym


class RewardClipper(gym.Wrapper):
    """
        A wrapper for reward clipping.
    """

    '''
        Arguments for the constructor:
        env: Game environment to be history stacking wrapped;
        lower: lower bound of reward; float;
        upper: upper bound of reward; float.
    '''
    def __init__(self, env, lower=-1.0, upper=1.0):
        super().__init__(env)
        self.lower = lower
        self.upper = upper

    def reset(self):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = min(max(reward, self.lower), self.upper)
        return obs, reward, done, info


