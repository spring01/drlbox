
import time
import numpy as np
import tensorflow as tf
from drlbox.common.util import set_args


class Evaluator:

    KEYWORD_DICT = dict(make_env=None,
                        state_to_input=None,
                        load_model=None,
                        render_timestep=None,
                        render_end=False,
                        num_episodes=20,)

    def __init__(self, **kwargs):
        set_args(self, self.KEYWORD_DICT, kwargs)

    def run(self):
        env = self.make_env()
        self.setup_algorithm(env.action_space)

        saved_model = self.net_cls.load_model(self.load_model)
        net = self.net_cls.from_model(saved_model)

        # global_variables_initializer will re-initialize net.weights
        # and so we need to sync to saved_weights
        saved_weights = saved_model.get_weights()
        sess = tf.Session()
        net.set_session(sess)
        sess.run(tf.global_variables_initializer())
        net.set_sync_weights(saved_weights)
        net.sync()

        # evaluation
        all_total_rewards = []
        for _ in range(self.num_episodes):
            state = env.reset()
            self.render_env_at_timestep(env)
            total_rewards = 0.0
            while True:
                state = self.state_to_input(state)
                action_values = net.action_values(np.stack([state]))[0]
                action = self.policy.select_action(action_values)
                state, reward, done, info = env.step(action)
                self.render_env_at_timestep(env)
                total_rewards += reward
                if done:
                    break
            if self.render_end:
                env.render()
            all_total_rewards.append(total_rewards)
            print('episode reward:', total_rewards)
        print('average episode reward:', np.mean(all_total_rewards))

    def render_env_at_timestep(self, env):
        if self.render_timestep is not None:
            env.render()
            time.sleep(self.render_timestep)


