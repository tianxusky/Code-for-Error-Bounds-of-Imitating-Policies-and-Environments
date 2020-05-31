__all__ = ['BaseBatchedEnv', 'DummyVecEnv', 'SubprocVecEnv']
import abc
import numpy as np
import gym
from multiprocessing import Process, Pipe


class BaseBatchedEnv(gym.Env, abc.ABC):
    # thought about using `@property @abc.abstractmethod` here but we don't need explicit `@property` function here.
    n_envs = None
    max_episode_steps = None

    @abc.abstractmethod
    def partial_reset(self, indices):
        pass

    @abc.abstractmethod
    def partial_step(self, indices, actions):
        pass


class VecEnv(abc.ABC):
    def __init__(self, num_envs, observation_space, action_space):
        self.n_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abc.abstractmethod
    def reset(self):
        pass

    @abc.abstractmethod
    def step_async(self, actions):
        pass

    @abc.abstractmethod
    def step_wait(self):
        pass

    @abc.abstractmethod
    def close(self):
        pass

    # @timeit
    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv, BaseBatchedEnv, gym.Wrapper):
    def __init__(self, env_fns):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)
        env = gym.make(env_fns[0]().unwrapped.spec.id)
        self.max_episode_steps = env._max_episode_steps
        env.close()

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def partial_reset(self, indices):
        for idx in indices:
            self.remotes[idx].send(('reset', None))
        return np.stack([self.remotes[idx].recv() for idx in indices])

    def partial_step(self, indices, actions):
        for e, idx in enumerate(indices):
            self.remotes[idx].send(('step', actions[e]))
        results = [self.remotes[idx].recv() for idx in indices]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True


class DummyVecEnv(VecEnv, BaseBatchedEnv, gym.Wrapper):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        self.buf_obs = np.zeros((self.n_envs,) + env.observation_space.shape, env.observation_space.dtype)
        self.buf_dones = np.zeros((self.n_envs,), dtype=np.bool)
        self.buf_rews = np.zeros((self.n_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.n_envs)]
        self.actions = None

        env = gym.make(env_fns[0]().unwrapped.spec.id)
        self.max_episode_steps = env._max_episode_steps
        env.close()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for i in range(self.n_envs):
            self.buf_obs[i], self.buf_rews[i], self.buf_dones[i], self.buf_infos[i] = self.envs[i].step(self.actions[i])
        return np.asarray(self.buf_obs), np.asarray(self.buf_rews), np.asarray(self.buf_dones), self.buf_infos

    def reset(self):
        for i in range(self.n_envs):
            self.buf_obs[i] = self.envs[i].reset()
        return np.asarray(self.buf_obs)

    def partial_reset(self, indices):
        return self.reset()

    def partial_step(self, indices, actions):
        return self.step(actions)

    def close(self):
        return

