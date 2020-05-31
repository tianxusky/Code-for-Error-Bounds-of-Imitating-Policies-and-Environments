import numpy as np
from lunzi.dataset import Dataset
import sys


class ReplayBuffer(object):
    def __init__(self, num_envs, n_steps, dtype, stacked_frame=False, size=50000):
        self.num_envs = num_envs
        self.n_steps = n_steps
        self.dtype = dtype
        self.stacked_frame = stacked_frame
        self._size = size // n_steps

        # Memory
        self.obs_shape, self.obs_dtype = None, None
        self.state_block = None
        self.actions = None
        self.rewards = None
        self.mus = None
        self.dones = None
        self.timeouts = None
        self.infos = None

        # Size indexes
        self._next_idx = 0
        self._total_size = 0
        self._num_in_buffer = 0

    # @timeit
    def store_episode(self, data: Dataset):
        data = data.reshape([self.n_steps, self.num_envs])

        if self.state_block is None:
            self.obs_shape, self.obs_dtype = list(data.state.shape[2:]), data.state.dtype
            self.state_block = np.empty([self._size], dtype=object)
            self.actions = np.empty([self._size] + list(data.action.shape), dtype=data.action.dtype)
            self.rewards = np.empty([self._size] + list(data.reward.shape), dtype=data.reward.dtype)
            self.mus = np.empty([self._size] + list(data.mu.shape), dtype=data.mu.dtype)
            self.dones = np.empty([self._size] + list(data.done.shape), dtype=np.bool)
            self.timeouts = np.empty([self._size] + list(data.timeout.shape), dtype=np.bool)
            self.infos = np.empty([self._size] + list(data.info.shape), dtype=object)

        terminals = data.done | data.timeout
        if self.stacked_frame:
            self.state_block[self._next_idx] = StackedFrame(data.state, data.next_state, terminals)
        else:
            self.state_block[self._next_idx] = StateBlock(data.state, data.next_state, terminals)
        self.actions[self._next_idx] = data.action
        self.rewards[self._next_idx] = data.reward
        self.mus[self._next_idx] = data.mu
        self.dones[self._next_idx] = data.done
        self.timeouts[self._next_idx] = data.timeout
        self.infos[self._next_idx] = data.info

        self._next_idx = (self._next_idx + 1) % self._size
        self._total_size += 1
        self._num_in_buffer = min(self._size, self._num_in_buffer + 1)

    # @timeit
    def sample(self, idx=None, envx=None):
        assert self.can_sample()
        idx = np.random.randint(self._num_in_buffer, size=self.num_envs) if idx is None else idx
        num_envs = self.num_envs

        envx = np.arange(num_envs) if envx is None else envx

        take = lambda x: self.take(x, idx, envx)  # for i in range(num_envs)], axis = 0)

        # (nstep, num_envs)
        states = self.take_block(self.state_block, idx, envx, 0)
        next_states = self.take_block(self.state_block, idx, envx, 1)
        actions = take(self.actions)
        mus = take(self.mus)
        rewards = take(self.rewards)
        dones = take(self.dones)
        timeouts = take(self.timeouts)
        infos = take(self.infos)

        samples = Dataset(dtype=self.dtype, max_size=self.num_envs*self.n_steps)
        steps = [states, actions, next_states, mus, rewards, dones, timeouts, infos]
        steps = list(map(flatten_first_2_dims, steps))
        samples.extend(np.rec.fromarrays(steps, dtype=self.dtype))
        return samples

    def take(self, x, idx, envx):
        num_envs = self.num_envs
        out = np.empty([self.n_steps, num_envs] + list(x.shape[3:]), dtype=x.dtype)
        for i in range(num_envs):
            out[:, i] = x[idx[i], :, envx[i]]
        return out

    def take_block(self, x, idx, envx, block_idx):
        num_envs = self.num_envs
        out = np.empty([self.n_steps, num_envs] + self.obs_shape, dtype=self.obs_dtype)
        for i in range(num_envs):
            if self.stacked_frame:
                out[:, i] = x[idx[i]].get(block_idx, envx[i])  # accelerate by specifying env_idx
            else:
                out[:, i] = x[idx[i]][block_idx][:, envx[i]]
        return out

    def can_sample(self):
        return self._num_in_buffer > 0

    def get_current_size(self):
        return self._num_in_buffer * self.num_envs * self.n_steps

    def get_cumulative_size(self):
        return self._total_size * self.num_envs * self.n_steps

    def iterator(self, batch_size, random=False):
        assert self._num_in_buffer >= batch_size
        indices = np.arange(self._next_idx-batch_size, self._next_idx) % self._size
        if random:
            np.random.shuffle(indices)
        for idx in indices:
            envx = np.arange(self.num_envs)
            next_states = self.take_block(self.state_block, [idx for _ in range(self.num_envs)], envx, 1)
            infos = self.take(self.infos, [idx for _ in range(self.num_envs)], envx)
            yield next_states, infos


class StateBlock(object):
    __slots__ = '_data', '_idx', '_append_value'

    def __init__(self, x, x2, done):
        nstep, num_envs = x.shape[:2]
        assert x2.shape[:2] == done.shape == (nstep, num_envs)
        _done = done.copy()
        _done[-1, :] = True
        self._idx = np.where(_done)
        self._append_value = x2[self._idx]
        self._data = x

    def __getitem__(self, index):
        assert index in {0, 1}
        if index == 0:
            return self._data
        else:
            x = np.roll(self._data, -1, axis=0)
            x[self._idx] = self._append_value
            return x

    def __sizeof__(self):
        return sys.getsizeof(self._idx) + sys.getsizeof(self._append_value) + sys.getsizeof(self._data)


class Frame:
    def __init__(self, x, x2, done):
        self._n_step, self._nh, self._nw, self._n_stack = x.shape
        assert x.shape == x2.shape and done.shape == (self._n_step, )
        frames = np.split(x[0], self._n_stack, axis=-1)
        for t in range(self._n_step):
            frames.append(x2[t, ..., -1][..., None])
            if t < self._n_step-1 and done[t]:
                frames.extend(np.split(x[t+1], self._n_stack, axis=-1))
        self._frames = frames
        self._idx = np.where(done)[0]

    def __getitem__(self, index):
        assert index in {0, 1}
        if index == 0:
            x = np.zeros([self._n_step, self._nh, self._nw, self._n_stack])
            x[0] = np.concatenate(self._frames[:self._n_stack], axis=-1)
            start = 1
            for t in range(1, self._n_step):
                if t-1 in self._idx:
                    start += self._n_stack
                x[t] = np.concatenate(self._frames[start:start+self._n_stack], axis=-1)
                start += 1
            return x
        else:
            x2 = np.zeros([self._n_step, self._nh, self._nw, self._n_stack])
            x2[0] = np.concatenate(self._frames[1:1+self._n_stack], axis=-1)
            start = 2
            for t in range(1, self._n_step):
                if t-1 in self._idx:
                    start += self._n_stack
                x2[t] = np.concatenate(self._frames[start:start+self._n_stack], axis=-1)
                start += 1
            return x2


class StackedFrame:
    def __init__(self, x, x2, done):
        n_step, self._n_env = x.shape[:2]
        assert x.shape == x2.shape and done.shape == (n_step, self._n_env)
        self._frames = [Frame(x[:, e], x2[:, e], done[:, e]) for e in range(self._n_env)]

    def get(self, index, env_idx=None):
        assert index in {0, 1}, 'index: %d should be 0 or 1' % index
        if env_idx is None:
            frames = [self._frames[e][index] for e in range(self._n_env)]
            return np.array(frames).swapaxes(1, 0)
        else:
            assert 0 <= env_idx < self._n_env, 'env_idx: %d should be less than num_env: %d' % (env_idx, self._n_env)
            return self._frames[env_idx][index]


def flatten_first_2_dims(x):
    return x.reshape([-1, *x.shape[2:]])


def test_stacked_frame():
    import time
    n_step, n_env, n_stack = 20, 2, 4
    frames = []
    for _ in range(n_step+n_stack):
        frames.append(np.random.randn(n_env, 84, 84))
    x = [np.stack(frames[:n_stack], axis=-1)]
    x2 = [np.stack(frames[1:1+n_stack], axis=-1)]
    for i in range(1, n_step):
        x.append(np.stack(frames[i:i+n_stack], axis=-1))
        x2.append(np.stack(frames[i+1: i+1+n_stack], axis=-1))
    x, x2 = np.array(x), np.array(x2)
    # print(x.shape, x2.shape)
    assert np.array_equal(x[1:], x2[:-1])
    done = np.zeros([n_step, n_env], dtype=bool)
    done[(np.random.randint(0, n_step, 3), np.random.randint(0, n_env, 3))] = True
    # print(np.where(done))
    ts = time.time()
    buf = StackedFrame(x, x2, done)
    print('new store time: %.3f sec' % (time.time() - ts))
    ts = time.time()
    x_, x2_ = buf.get(0), buf.get(1)
    print('new sample time:%.3f sec' % (time.time() - ts))

    ts = time.time()
    buf_ref = StateBlock(x, x2, done)
    print('old store time: %.3f sec' % (time.time() - ts))
    ts = time.time()
    x_ref, x2_ref = buf_ref[0], buf_ref[1]
    print('old sample time:%.3f sec' % (time.time() - ts))

    np.testing.assert_allclose(x_, x_ref)
    np.testing.assert_allclose(x2_, x2_ref)
    for e in range(n_env):
        for t in range(n_step):
            np.testing.assert_allclose(x[t, e], x_[t, e], err_msg='t=%d, e=%d' % (t, e))
            np.testing.assert_allclose(x2[t, e], x2_[t, e], err_msg='t=%d, e=%d' % (t, e))


if __name__ == '__main__':
    for _ in range(10):
        test_stacked_frame()
