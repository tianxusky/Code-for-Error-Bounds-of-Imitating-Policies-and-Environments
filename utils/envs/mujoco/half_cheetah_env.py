# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np
from gym.envs.mujoco import half_cheetah
from .virtual_env import BaseModelBasedEnv


class HalfCheetahEnv(half_cheetah.HalfCheetahEnv, BaseModelBasedEnv):
    scaling = 10.
    absolute_reward = False

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.1 * np.square(action).sum()
        if self.absolute_reward:
            reward_run = xposafter / self.scaling * 0.5
        else:
            reward_run = (xposafter - xposbefore)/self.dt
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[0][None] / self.scaling,
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def mb_step(self, states, actions, next_states):
        xposbefore = states[:, 0]
        xposafter = next_states[:, 0]
        reward_ctrl = - 0.1 * np.sum(np.square(actions), axis=-1)
        if self.absolute_reward:
            reward_run = xposafter * 0.5
        else:
            reward_run = (xposafter - xposbefore)/self.dt * self.scaling
        rewards = reward_ctrl + reward_run
        dones = np.zeros(states.shape[0], dtype=np.bool)
        return rewards, dones
