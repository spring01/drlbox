
import time
import tensorflow as tf
from numpy import mean
from drlbox.common.util import set_args
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import StochasticDisc, StochasticCont, EpsGreedy


class Evaluator:

    KEYWORD_DICT = dict(env_maker=None,
                        state_to_input=None,
                        load_model=None,
                        render_timestep=None,
                        render_end=False,
                        num_episodes=20,
                        policy_type='stochastic',
                        policy_sto_cont_min_var=1e-4,
                        policy_eps=0.0,)

    def __init__(self, **kwargs):
        set_args(self, self.KEYWORD_DICT, kwargs)

    def run(self):
        env = self.env_maker()

        # setup policy
        self.policy_type = self.policy_type.lower()
        if self.policy_type == 'stochastic':
            if discrete_action(env.action_space):
                self.policy = StochasticDisc()
            elif continuous_action(env.action_space):
                self.policy = StochasticCont(low=env.action_space.low,
                    high=env.action_space.high,
                    min_var=self.policy_sto_cont_min_var)
            else:
                raise TypeError('Type of action_space not valid')
        elif self.policy_type == 'greedy':
            if not discrete_action(env.action_space):
                raise TypeError('greedy policy supports only discrete action.')
            self.policy = EpsGreedy(self.policy_eps)
        else:
            raise ValueError('policy type {} invalid.'.format(self.policy_type))

        # load model
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
                action_values = net.action_values([state])[0]
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
        print('average episode reward:', mean(all_total_rewards))

    def render_env_at_timestep(self, env):
        if self.render_timestep is not None:
            env.render()
            time.sleep(self.render_timestep)


