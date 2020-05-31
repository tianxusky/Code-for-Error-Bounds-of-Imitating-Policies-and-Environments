from gym import spaces
import numpy as np
from .virtual_env import BaseModelBasedEnv


def initializer(shape, loc=0., scale=np.sqrt(2), dtype=np.float32, svd=True, np_random=None):
    shape = tuple(shape)
    assert len(shape) == 2
    a = np_random.normal(loc, 1.0, shape)
    if svd:
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == shape else v  # pick the one with the correct shape
        q = q.reshape(shape)
    else:
        q = a
    return (scale * q[:shape[0], :shape[1]]).astype(dtype)


class Actor(object):
    def __init__(self, dim_state, dim_action, init_std=0., random_state=1516):
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.init_std = init_std

        self.np_random = np.random.RandomState(random_state)
        self.parameter = initializer([dim_state, dim_action], loc=0., scale=1., np_random=self.np_random)
        self.noise = initializer([dim_state, dim_action], loc=0., scale=init_std, np_random=self.np_random)

    def get_actions(self, state_, fetch='actions'):
        state_ = state_.copy()
        if state_.ndim == 1:
            state_ = state_[None, :]
        if fetch == 'actions_mean':
            action_ = np.dot(state_, self.parameter)
            action_ = np.sin(action_)
        elif fetch == 'actions':
            action_ = np.dot(state_, self.parameter + self.noise)
            action_ = np.sin(action_)
        else:
            raise ValueError('fetch = %s is not supported' % fetch)
        return action_


class LinearEnv(BaseModelBasedEnv):
    def __init__(self, dim_state=5, dim_action=3, noise=0.05, max_episode_steps=1000,
                 random_state=2020):
        self.np_random = np.random.RandomState(seed=random_state)
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.noise = noise
        self.max_episode_steps = max_episode_steps

        self.w = self.np_random.normal(loc=0., scale=1., size=[dim_state + dim_action, dim_state])
        self.b = self.np_random.normal(loc=0., scale=0.1, size=[dim_state])

        self.add_dim = add_dim = 2

        def forward_predict(state_, action_):
            input_ = np.concatenate([state_, action_], axis=1)
            h1 = np.dot(input_, self.w) + self.b
            output_ = np.sin(h1)
            if state_.ndim == 1:
                output_[:add_dim] = np.abs(output_[:add_dim]) * 0.01
                output_[:add_dim] += state_[:add_dim]
            else:
                output_[:, :add_dim] = np.abs(output_[:, :add_dim]) * 0.01
                output_[:, :add_dim] += state_[:, :add_dim]
            return output_

        self.forward_fn = forward_predict

        self.action_space = spaces.Box(low=-1., high=1., shape=[self.dim_action])
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=[self.dim_state])

        self.state = None
        self.nb_step = 0
        self.rewarded_policy = Actor(dim_state, dim_action, init_std=0., random_state=1516)
        self.error_bound = 0.2

    def step(self, action: np.ndarray):
        self.nb_step += 1
        assert action.shape == (self.dim_action, )
        next_state = self.forward_fn(self.state[None], action[None])[0] + np.random.randn(self.dim_state) * self.noise

        reward_action = self.rewarded_policy.get_actions(self.state[None], fetch='actions_mean')
        reward_next_state = self.forward_fn(self.state[None], reward_action)[0]
        reward = -np.linalg.norm(reward_next_state - next_state) + 1

        error = np.linalg.norm(reward_next_state[self.add_dim:] - next_state[self.add_dim:], ord=np.inf)
        done = error > self.error_bound
        self.state = next_state
        return next_state, reward, done, {}

    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        if states.ndim == 1:
            states = states[None, :]
        
        reward_actions = self.rewarded_policy.get_actions(states, fetch='actions_mean')
        reward_next_states = self.forward_fn(states, reward_actions)
        rewards = -np.linalg.norm(reward_next_states - next_states, axis=1) + 1
        errors = np.linalg.norm(reward_next_states[:, self.add_dim:] - next_states[:, self.add_dim:], axis=1, ord=np.inf)
        dones = errors > self.error_bound
        return rewards, dones

    def reset(self):
        self.nb_step = 0
        self.state = self.np_random.uniform(low=0, high=0.1, size=[self.dim_state])
        return self.state.copy()


if __name__ == '__main__':
    env = LinearEnv()
    actor = Actor(env.dim_state, env.dim_action)

    state_list, action_list = [], []
    for _ in range(10):
        state = env.reset()
        return_ = 0
        print(state[0])
        for i in range(env.max_episode_steps):
            action = actor.get_actions(state)[0]
            next_state, reward, done, info = env.step(action)
            return_ += reward
            state_list.append(next_state)
            action_list.append(action)
            if done:
                break
            state = next_state
        print(i, state[0], return_)
        # assert 0
    state_list = np.array(state_list)
    action_list = np.array(action_list)

    print('mean:', np.mean(state_list, axis=0), '\nstd:', np.std(state_list, axis=0))
    print('mean:', np.mean(action_list, axis=0), '\nstd:', np.std(action_list, axis=0))
