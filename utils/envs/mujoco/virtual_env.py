# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import abc
import gym
from lunzi.Logger import logger
from trpo.utils.runner import Dataset, gen_dtype


class BaseBatchedEnv(gym.Env, abc.ABC):
    # thought about using `@property @abc.abstractmethod` here but we don't need explicit `@property` function here.
    n_envs: int

    @abc.abstractmethod
    def step(self, actions):
        pass

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass

    def set_state(self, state):
        logger.warning('`set_state` is not implemented')


class BaseModelBasedEnv(gym.Env, abc.ABC):
    @abc.abstractmethod
    def mb_step(self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray):
        raise NotImplementedError

    def verify(self, n=10000, eps=1e-4):
        dataset = Dataset(gen_dtype(self, 'state action next_state reward done'), n)
        state = self.reset()
        for _ in range(n):
            action = self.action_space.sample()
            next_state, reward, done, _ = self.step(action)
            dataset.append((state, action, next_state, reward, done))

            state = next_state
            if done:
                state = self.reset()

        rewards_, dones_ = self.mb_step(dataset.state, dataset.action, dataset.next_state)
        diff = dataset.reward - rewards_
        l_inf = np.abs(diff).max()
        logger.warning('rewarder difference: %.6f', l_inf)

        np.testing.assert_allclose(dones_, dataset.done)
        assert not np.isclose(np.std(dataset.state, axis=0), 0.).any(), \
            'state.std:{}'.format(np.std(dataset.state, axis=0))
        assert l_inf < eps


class VirtualEnv(BaseBatchedEnv):
    _states: np.ndarray

    def __init__(self, model, env: BaseModelBasedEnv, n_envs: int, opt_model=False,
                 stochastic_model=False):
        super().__init__()
        self.n_envs = n_envs
        self.observation_space = env.observation_space  # ???

        dim_state = env.observation_space.shape[0]
        dim_action = env.action_space.shape[0]
        if opt_model:
            self.action_space = gym.spaces.Box(
                low=np.r_[env.action_space.low, np.zeros(dim_state) - 1.],
                high=np.r_[env.action_space.high, np.zeros(dim_state) + 1.],
                dtype=np.float32
            )
        else:
            self.action_space = env.action_space

        self._opt_model = opt_model
        self._stochastic_model = stochastic_model
        self._model = model
        self._env = env

        self._states = np.zeros((self.n_envs, dim_state), dtype=np.float32)

    def _scale_action(self, actions):
        lo, hi = self.action_space.low, self.action_space.high
        return lo + (actions + 1.) * 0.5 * (hi - lo)

    def step(self, actions):
        if self._opt_model:
            actions = actions[..., :self._env.action_space.shape[0]]

        if self._stochastic_model:
            next_states = self._model.eval('next_states', states=self._states, actions=actions)
        else:
            next_states = self._model.eval('next_states_mean', states=self._states, actions=actions)
        rewards, dones = self._env.mb_step(self._states, self._scale_action(actions), next_states)

        self._states = next_states
        return self._states.copy(), rewards, dones, [{} for _ in range(self.n_envs)]

    def reset(self):
        return self.partial_reset(range(self.n_envs))

    def partial_reset(self, indices):
        initial_states = np.array([self._env.reset() for _ in indices])

        self._states = self._states.copy()
        self._states[indices] = initial_states

        return initial_states.copy()

    def set_state(self, states):
        self._states = states.copy()

    def render(self, mode='human'):
        pass
