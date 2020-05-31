import gym
import numpy as np
from lunzi.dataset import Dataset
from ..policies import BaseNNPolicy
from utils.envs.batched_env import BaseBatchedEnv


class Runner(object):
    def __init__(self, env: BaseBatchedEnv, max_steps: int, gamma=0.99):
        self.env = env
        self.n_envs = env.n_envs
        self.gamma = gamma
        self.max_steps = max_steps
        self._dtype = gen_dtype(env, 'state action next_state mu reward done timeout info nstep')

        self.reset()

    def reset(self):
        self._states = self.env.reset()
        self._n_steps = np.zeros(self.n_envs, 'i4')
        self._returns = np.zeros(self.n_envs, 'f8')

    def run(self, policy: BaseNNPolicy, n_steps: int, stochastic=True):
        ep_infos = []
        n_samples = n_steps * self.n_envs
        dataset = Dataset(self._dtype, n_samples)

        for T in range(n_steps):
            if stochastic:
                actions, mus = policy.get_actions(self._states, fetch='actions mus')
            else:
                actions, mus = policy.get_actions(self._states, fetch='actions_mean mus')

            next_states, rewards, dones, infos = self.env_step(actions, mus)
            dones = dones.astype(bool)
            timeouts = self._n_steps == self.max_steps

            steps = [self._states.copy(), actions, next_states, mus, rewards, dones, timeouts, infos, self._n_steps.copy()]
            dataset.extend(np.rec.fromarrays(steps, dtype=self._dtype))

            indices = np.where(dones | timeouts)[0]
            if len(indices) > 0:
                new_states = self.env.partial_reset(indices)
                for e, index in enumerate(indices):
                    next_states[index] = new_states[e]
                    infos[index]['episode'] = {'return': self._returns[index], 'length': self._n_steps[index]}
                self._n_steps[indices] = 0
                self._returns[indices] = 0.

            self._states = next_states
            ep_infos.extend([info['episode'] for info in infos if 'episode' in info])

        return dataset, ep_infos

    def env_step(self, actions, mus):
        next_states, rewards, dones, infos = self.env.step(actions)
        self._returns += rewards
        self._n_steps += 1
        return next_states, rewards, dones, infos

    def compute_qret(self, policy: BaseNNPolicy, samples: Dataset):
        n_steps = len(samples) // self.n_envs
        q_is, vs, mus = policy.get_q_values(samples.state, samples.action, fetch='q_values_ v_values mus')
        rho = np.divide(mus, samples.mu + 1e-6)
        rho_i = get_by_index(rho, samples.action)
        rho_bar = np.minimum(1.0, rho_i)
        rho_bar = rho_bar.reshape((n_steps, self.n_envs))
        q_is = q_is.reshape((n_steps, self.n_envs))
        vs = vs.reshape((n_steps, self.n_envs))
        samples = samples.reshape((n_steps, self.n_envs))
        terminals = samples.done | samples.timeout
        next_values = policy.get_v_values(samples[-1].next_state)

        qret = next_values
        qrets = []
        for i in range(n_steps - 1, -1, -1):
            qret = samples.reward[i] + self.gamma * qret * (1.0 - terminals[i])
            qrets.append(qret)
            qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
        qrets = qrets[::-1]
        qret = np.array(qrets, dtype='f8')
        qret = np.reshape(qret, [-1])
        return qret


def get_by_index(x, index):
    assert x.ndim == 2 and len(index) == len(x)
    indices = np.arange(len(x))
    return x[(indices, index)]


def gen_dtype(env: gym.Env, fields: str):
    dtypes = {
        'state': ('state', env.observation_space.dtype, env.observation_space.shape),
        'action': ('action', env.action_space.dtype, env.action_space.shape),
        'next_state': ('next_state', env.observation_space.dtype, env.observation_space.shape),
        'reward': ('reward', 'f8'),
        'done': ('done', 'bool'),
        'timeout': ('timeout', 'bool'),
        'qret': ('qret', 'f8'),
        'mu': ('mu', 'f8', (env.action_space.n, )),
        'nstep': ('nstep', 'i4',),
        'info': ('info', object)
    }
    return [dtypes[field] for field in fields.split(' ')]


if __name__ == '__main__':
    import tensorflow as tf


    def seq_to_batch(h, flat=False):
        shape = h[0].get_shape().as_list()
        if not flat:
            assert (len(shape) > 1)
            nh = h[0].get_shape()[-1].value
            return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
        else:
            return tf.reshape(tf.stack(values=h, axis=1), [-1])

    # remove last step
    def strip(var, nenvs, nsteps, flat=False):
        vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
        return seq_to_batch(vars[:-1], flat)


    def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
        """
        Calculates q_retrace targets

        :param R: Rewards
        :param D: Dones
        :param q_i: Q values for actions taken
        :param v: V values
        :param rho_i: Importance weight for each action
        :return: Q_retrace values
        """
        rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
        rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
        ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
        q_is = batch_to_seq(q_i, nenvs, nsteps, True)
        vs = batch_to_seq(v, nenvs, nsteps + 1, True)
        v_final = vs[-1]
        qret = v_final
        qrets = []
        for i in range(nsteps - 1, -1, -1):
            check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
            qret = rs[i] + gamma * qret * (1.0 - ds[i])
            qrets.append(qret)
            qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
        qrets = qrets[::-1]
        qret = seq_to_batch(qrets, flat=True)
        return qret


    def batch_to_seq(h, nbatch, nsteps, flat=False):
        if flat:
            h = tf.reshape(h, [nbatch, nsteps])
        else:
            h = tf.reshape(h, [nbatch, nsteps, -1])
        return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]

