# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
import gym
from utils.envs.batched_env import BaseBatchedEnv
from trpo.policies import BasePolicy
from trpo.v_function import BaseVFunction
from lunzi.dataset import Dataset


class Runner(object):
    _states: np.ndarray  # [np.float]
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env: BaseBatchedEnv, max_steps: int, gamma=0.99, lambda_=0.95, rescale_action=False,
                 partial_episode_bootstrapping=False):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.rescale_action = rescale_action
        self.partial_episode_bootstrapping = partial_episode_bootstrapping
        self._dtype = gen_dtype(env, 'state action next_state reward done timeout')

        self.reset()

    def reset(self):
        self._states = self.env.reset()
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def run(self, policy: BasePolicy, n_samples: int):
        ep_infos = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples)

        for T in range(n_steps):
            unscaled_actions = policy.get_actions(self._states)
            if self.rescale_action:
                lo, hi = self.env.action_space.low, self.env.action_space.high
                actions = lo + (unscaled_actions + 1.) * 0.5 * (hi - lo)
            else:
                actions = unscaled_actions

            next_states, rewards, dones, infos = self.env.step(actions)
            dones = dones.astype(bool)
            self._returns += rewards
            self._n_steps += 1
            timeouts = self._n_steps == self.max_steps
            terminals = np.copy(dones)
            for e, info in enumerate(infos):
                if self.partial_episode_bootstrapping and info.get('TimeLimit.truncated', False):
                    terminals[e] = False

            steps = [self._states.copy(), unscaled_actions, next_states.copy(), rewards, terminals, timeouts]
            dataset.extend(np.rec.fromarrays(steps, dtype=self._dtype))

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                next_states = next_states.copy()
                next_states[indices] = self.env.partial_reset(indices)
                for index in indices:
                    infos[index]['episode'] = {'return': self._returns[index], 'length': self._n_steps[index]}
                self._n_steps[indices] = 0
                self._returns[indices] = 0.

            self._states = next_states.copy()
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        return dataset, ep_infos

    def compute_advantage(self, vfn: BaseVFunction, samples: Dataset):
        n_steps = len(samples) // self.n_envs
        samples = samples.reshape((n_steps, self.n_envs))
        use_next_vf = ~samples.done
        use_next_adv = ~(samples.done | samples.timeout)

        next_values = vfn.get_values(samples.reshape(-1).next_state).reshape(n_steps, self.n_envs)
        values = vfn.get_values(samples.reshape(-1).state).reshape(n_steps, self.n_envs)
        advantages = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        last_gae_lambda = 0

        for t in reversed(range(n_steps)):
            delta = samples[t].reward + self.gamma * next_values[t] * use_next_vf[t] - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda * use_next_adv[t]
        return advantages.reshape(-1), values.reshape(-1)


def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        'state': ('state', env.observation_space.dtype, env.observation_space.shape),
        'action': ('action', env.action_space.dtype, env.action_space.shape),
        'next_state': ('next_state', env.observation_space.dtype, env.observation_space.shape),
        'reward': ('reward', 'f8'),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
    }
    return [dtypes[field] for field in fields.split(' ')]


def evaluate(policy, env, num_episodes=10, deterministic=True):
    total_returns = []
    total_lengths = []
    total_episodes = 0

    n_returns = np.zeros(env.n_envs)
    n_lengths = np.zeros(env.n_envs)
    states = env.reset()
    while total_episodes < num_episodes:
        if deterministic:
            actions = policy.get_actions(states, fetch='actions_mean')
        else:
            actions = policy.get_actions(states)
        next_states, rewards, dones, _ = env.step(actions)
        n_returns += rewards
        n_lengths += 1
        indices = np.where(dones)[0]
        if len(indices) > 0:
            next_states[indices] = env.partial_reset(indices)
            total_returns.extend(n_returns[indices].copy())
            total_lengths.extend(n_lengths[indices].copy())
            total_episodes += np.sum(indices)
            n_returns[indices] = 0
            n_lengths[indices] = 0
        states = next_states

    return total_returns, total_lengths
