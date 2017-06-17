
import gym

class HistoryStacker(gym.Wrapper):
    """
        A wrapper for history stacking.
        Stacks `num_frames` most recent frames together,
        and performs action for `act_steps` steps each time env.step is called.
    """

    '''
        Arguments for the constructor:
        env: Game environment to be history stacking wrapped;
        num_frames: Number of frames to be stacked together; integer;
        act_steps: Number of actions performed between states; integer.
    '''
    def __init__(self, env, num_frames=4, act_steps=2):
        super(HistoryStacker, self).__init__(env)
        obs_space = tuple([env.observation_space] * num_frames)
        self.observation_space = gym.spaces.Tuple(obs_space)
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
        return tuple(self.obs_list), total_reward, done, info

    def _reset(self):
        obs = self.env.reset()
        self.obs_list = [obs] * self.num_frames
        return tuple(self.obs_list)



