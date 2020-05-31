# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import gym
from trpo.policies import BasePolicy
from trpo.v_function import BaseVFunction
from lunzi.dataset import Dataset
from .replay_buffer import Mask


class Runner(object):
    _states: np.ndarray  # [np.float]
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env, max_steps: int, gamma=0.99, lambda_=0.95, rescale_action=False, add_absorbing_state=False):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.rescale_action = rescale_action
        self.add_absorbing_state = add_absorbing_state
        self._dtype = gen_dtype(env, 'state action next_state reward done timeout mask step')

        self.reset()

    def reset(self):
        self._state = self.env.reset()
        self._n_step = 0
        self._return = 0

    def run(self, policy: BasePolicy, n_samples: int, stochastic=True):
        assert self.n_envs == 1, 'Only support 1 env.'
        ep_infos = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples)

        for t in range(n_samples):
            if stochastic:
                unscaled_action = policy.get_actions(self._state[None])[0]
            else:
                unscaled_action = policy.get_actions(self._state[None], fetch='actions_mean')[0]
            if self.rescale_action:
                lo, hi = self.env.action_space.low, self.env.action_space.high
                action = lo + (unscaled_action + 1.) * 0.5 * (hi - lo)
            else:
                action = unscaled_action

            next_state, reward, done, info = self.env.step(action)
            self._return += reward
            self._n_step += 1
            timeout = self._n_step == self.max_steps
            if not done or timeout:
                mask = Mask.NOT_DONE.value
            else:
                mask = Mask.DONE.value

            if self.add_absorbing_state and done and self._n_step < self.max_steps:
                next_state = self.env.get_absorbing_state()
            steps = [self._state.copy(), unscaled_action, next_state.copy(), reward, done, timeout, mask,
                     np.copy(self._n_step)]
            dataset.append(np.rec.array(steps, dtype=self._dtype))

            if done | timeout:
                if self.add_absorbing_state and self._n_step < self.max_steps:
                    action = np.zeros(self.env.action_space.shape)
                    absorbing_state = self.env.get_absorbing_state()
                    steps = [absorbing_state, action, absorbing_state, 0.0, False, False, Mask.ABSORBING.value]
                    dataset.append(np.rec.array(steps, dtype=self._dtype))
                    # t += 1
                next_state = self.env.reset()
                ep_infos.append({'return': self._return, 'length': self._n_step})
                self._n_step = 0
                self._return = 0.
            self._state = next_state.copy()

        return dataset, ep_infos

    def compute_advantage(self, vfn: BaseVFunction, samples: Dataset):
        n_steps = len(samples) // self.n_envs
        samples = samples.reshape((n_steps, self.n_envs))
        if not self.add_absorbing_state:
            use_next_vf = ~samples.done
            use_next_adv = ~(samples.done | samples.timeout)
        else:
            absorbing_mask = samples.mask == Mask.ABSORBING
            use_next_vf = np.ones_like(samples.done)
            use_next_adv = ~(absorbing_mask | samples.timeout)

        next_values = vfn.get_values(samples.reshape(-1).next_state).reshape(n_steps, self.n_envs)
        values = vfn.get_values(samples.reshape(-1).state).reshape(n_steps, self.n_envs)
        advantages = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        last_gae_lambda = 0

        for t in reversed(range(n_steps)):
            delta = samples[t].reward + self.gamma * next_values[t] * use_next_vf[t] - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda * use_next_adv[t]
            # next_values = values[t]
        return advantages.reshape(-1), values.reshape(-1)


def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        'state': ('state', env.observation_space.dtype, env.observation_space.shape),
        'action': ('action', env.action_space.dtype, env.action_space.shape),
        'next_state': ('next_state', env.observation_space.dtype, env.observation_space.shape),
        'reward': ('reward', 'f8'),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'mask': ('mask', 'i4'),
        'step': ('step', 'i8')
    }
    return [dtypes[field] for field in fields.split(' ')]


def evaluate(policy, env, num_episodes=10, gamma=1.0, deterministic=True):
    if hasattr(env, 'n_envs'):
        assert env.n_envs == 1

    total_returns = []
    total_lengths = []
    total_episodes = 0

    n_return = 0
    n_length = 0
    discount = 1.
    state = env.reset()
    while total_episodes < num_episodes:
        if deterministic:
            action = policy.get_actions(state[None], fetch='actions_mean')[0]
        else:
            action = policy.get_actions(state[None])[0]
        next_state, reward, done, _ = env.step(action)
        n_return += reward * discount
        discount *= gamma
        n_length += 1
        if done > 0:
            next_state = env.reset()
            total_returns.append(float(n_return))
            total_lengths.append(n_length)
            total_episodes += 1
            n_return = 0
            n_length = 0
            discount = 1.
        state = next_state

    return total_returns, total_lengths

