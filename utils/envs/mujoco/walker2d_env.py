# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from gym.envs.mujoco import walker2d
from .virtual_env import BaseModelBasedEnv


class Walker2dEnv(walker2d.Walker2dEnv, BaseModelBasedEnv):
    scaling = 10.
    absolute_reward = False

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        if self.absolute_reward:
            reward = posafter
        else:
            reward = ((posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and  ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0][None]/self.scaling,
            self.sim.data.qpos[1:],
            np.clip(self.sim.data.qvel, -10, 10)]).ravel()

    def mb_step(self, states, actions, next_states):
        posbefore = states[:, 0]
        posafter = next_states[:, 0]
        height = next_states[:, 1]
        ang = next_states[:, 2]
        alive_bonus = 1.0
        if self.absolute_reward:
            rewards = posafter * self.scaling
        else:
            rewards = ((posafter - posbefore) / self.dt) * self.scaling
        rewards += alive_bonus
        rewards -= 1e-3 * np.sum(np.square(actions), axis=-1)
        dones = ~ ((height > 0.8) & (height < 2.0) & (ang > -1.0) & (ang < 1.0))
        return rewards, dones
