# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from gym.envs.mujoco import hopper
from .virtual_env import BaseModelBasedEnv


class HopperEnv(hopper.HopperEnv, BaseModelBasedEnv):
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
            reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0][None] / self.scaling,
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def mb_step(self, states, actions, next_states):
        posbefore = states[:, 0]
        posafter = next_states[:, 0]
        alive_bonus = 1.0
        if self.absolute_reward:
            reward = posafter * self.scaling
        else:
            reward = (posafter - posbefore) / self.dt * self.scaling
        reward += alive_bonus
        reward -= 1e-3 * np.sum(np.square(actions), axis=-1)
        done = ~(
            (np.abs(next_states[:, 2:]) < 100).all(axis=-1) &
            (next_states[:, 1] > .7) &
            (np.abs(next_states[:, 2]) < .2)
        )
        return reward, done
