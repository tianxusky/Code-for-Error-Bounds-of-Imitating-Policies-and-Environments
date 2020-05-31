# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import gym
import numpy as np
from lunzi.dataset import Dataset
from sac.policies.actor import Actor
from mbrl.gail.v_function import BaseVFunction
from utils.envs.mujoco.virtual_env import VirtualEnv


class VirtualRunner(object):
    """Runner for GAIL"""
    _states: np.ndarray  # [np.float]
    _actions: np.ndarray
    _n_steps: np.ndarray
    _returns: np.ndarray

    def __init__(self, env: VirtualEnv, max_steps: int, gamma=0.99, lambda_=0.95, rescale_action=False):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.lambda_ = lambda_
        self.max_steps = max_steps
        self.rescale_action = rescale_action
        self._dtype = gen_dtype(env, 'state action next_state next_action reward done timeout step')

        self.reset()

    def reset(self):
        self.set_state(self.env.reset(), set_env_state=False)

    def set_state(self, states: np.ndarray, set_env_state=True):
        self._states = states.copy()
        self._actions = None
        if set_env_state:
            self.env.set_state(states)
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def get_state(self):
        return self._states.copy()

    def run(self, policy: Actor, n_samples: int, classifier=None, stochastic=True):
        ep_infos = []
        n_steps = n_samples // self.n_envs
        assert n_steps * self.n_envs == n_samples
        dataset = Dataset(self._dtype, n_samples)

        if self._actions is None:
            self._actions = self._get_action(policy, self._states, stochastic)
        for T in range(n_steps):
            unscaled_actions = self._actions.copy()
            if self.rescale_action:
                lo, hi = self.env.action_space.low, self.env.action_space.high
                actions = (lo + (unscaled_actions + 1.) * 0.5 * (hi - lo))
            else:
                actions = unscaled_actions

            next_states, rewards, dones, infos = self.env.step(actions)
            if classifier is not None:
                rewards = classifier.get_rewards(self._states, unscaled_actions, next_states)
            next_actions = self._get_action(policy, next_states, stochastic)
            dones = dones.astype(bool)
            self._returns += rewards
            self._n_steps += 1
            timeouts = self._n_steps == self.max_steps

            steps = [self._states.copy(), unscaled_actions, next_states.copy(), next_actions.copy(),
                     rewards, dones, timeouts, self._n_steps.copy()]
            dataset.extend(np.rec.fromarrays(steps, dtype=self._dtype))

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                next_states = next_states.copy()
                next_states[indices] = self.env.partial_reset(indices)
                next_actions = next_actions.copy()
                next_actions[indices] = self._get_action(policy, next_states, stochastic)[indices]
                for index in indices:
                    infos[index]['episode'] = {'return': self._returns[index], 'length': self._n_steps[index]}
                self._n_steps[indices] = 0
                self._returns[indices] = 0.

            self._states = next_states.copy()
            self._actions = next_actions.copy()
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        return dataset, ep_infos

    def compute_advantage(self, vfn: BaseVFunction, samples: Dataset):
        n_steps = len(samples) // self.n_envs
        samples = samples.reshape((n_steps, self.n_envs))
        use_next_vf = ~(samples.done | samples.timeout)
        use_next_adv = ~(samples.done | samples.timeout)

        next_values = vfn.get_values(samples[-1].next_state, samples[-1].next_action)
        values = vfn.get_values(samples.reshape(-1).state, samples.reshape(-1).action).reshape(n_steps, self.n_envs)
        advantages = np.zeros((n_steps, self.n_envs), dtype=np.float32)
        last_gae_lambda = 0

        for t in reversed(range(n_steps)):
            delta = samples[t].reward + self.gamma * next_values * use_next_vf[t] - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_ * last_gae_lambda * use_next_adv[t]
            next_values = values[t]
        return advantages.reshape(-1), values.reshape(-1)

    @staticmethod
    def _get_action(policy, states, stochastic):
        if stochastic:
            unscaled_actions = policy.get_actions(states)
        else:
            unscaled_actions = policy.get_actions(states, fetch='actions_mean')
        return unscaled_actions


def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        'state': ('state', env.observation_space.dtype, env.observation_space.shape),
        'action': ('action', env.action_space.dtype, env.action_space.shape),
        'next_state': ('next_state', env.observation_space.dtype, env.observation_space.shape),
        'next_action': ('next_action', env.action_space.dtype, env.action_space.shape),
        'reward': ('reward', 'f8'),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'step': ('step', 'i8'),
    }
    return [dtypes[field] for field in fields.split(' ')]


def evaluate(policy, env, num_episodes=10, gamma=1.0, deterministic=True, max_episode_steps=1000):
    total_returns = []
    total_lengths = []
    total_episodes = 0

    n_returns = np.zeros(env.n_envs, dtype=np.float32)
    n_lengths = np.zeros(env.n_envs, dtype=np.int32)
    discounts = np.ones(env.n_envs, dtype=np.float32)
    states = env.reset()
    while total_episodes < num_episodes:
        if deterministic:
            actions = policy.get_actions(states, fetch='actions_mean')
        else:
            actions = policy.get_actions(states)
        next_states, rewards, dones, _ = env.step(actions)
        n_returns += rewards * discounts
        discounts *= gamma
        n_lengths += 1

        timeouts = n_lengths == max_episode_steps
        indices = np.where(timeouts | dones)[0]
        if len(indices) > 0:
            np.testing.assert_allclose(rewards, env._env.mb_step(states, actions, next_states)[0], atol=1e-4, rtol=1e-4)
            next_states[indices] = env.partial_reset(indices)
            total_returns.extend(list(map(float, n_returns[indices])))
            total_lengths.extend(list(map(int, n_lengths[indices])))
            total_episodes += len(indices)
            n_returns[indices] = 0.
            n_lengths[indices] = 0
            discounts[indices] = 1.
        states = next_states

    return total_returns, total_lengths
