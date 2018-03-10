
import time
import tensorflow as tf
from drlbox.common.tasker import Tasker
from drlbox.common.util import discrete_action, continuous_action
from drlbox.common.policy import SoftmaxPolicy, GaussianPolicy, EpsGreedyPolicy


EVALUATOR_KWARGS = dict(
    render_timestep=None,
    render_end=False,
    num_episodes=20,
    policy_type='stochastic',
    policy_sto_cont_min_var=1e-4,
    policy_eps=0.0,
    )

class Evaluator(Tasker):

    KWARGS = {**Tasker.KWARGS, **EVALUATOR_KWARGS}

    def run(self):
        assert callable(self.env_maker)
        env = self.env_maker()

        # setup policy
        self.policy_type = self.policy_type.lower()
        if self.policy_type == 'stochastic':
            if discrete_action(env.action_space):
                self.policy = SoftmaxPolicy()
            elif continuous_action(env.action_space):
                self.policy = GaussianPolicy(
                    low=env.action_space.low,
                    high=env.action_space.high,
                    min_var=self.policy_sto_cont_min_var
                    )
            else:
                raise TypeError('Type of action_space not valid')
        elif self.policy_type == 'greedy':
            if not discrete_action(env.action_space):
                raise TypeError('greedy policy supports only discrete action.')
            self.policy = EpsGreedyPolicy(self.policy_eps)
        else:
            raise ValueError('policy type {} invalid.'.format(self.policy_type))

        # load model
        saved_model = self.do_load_model()
        net = self.net_cls()
        net.set_model(saved_model)

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
            self.print('episode reward: {}'.format(total_rewards))
        average_reward = sum(all_total_rewards) / len(all_total_rewards)
        self.print('average episode reward: {}'.format(average_reward))

    def render_env_at_timestep(self, env):
        if self.render_timestep is not None:
            env.render()
            time.sleep(self.render_timestep)


