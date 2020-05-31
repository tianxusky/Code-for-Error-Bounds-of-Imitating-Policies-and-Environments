from lunzi.Logger import logger
import numpy as np
from trpo.utils.normalizer import GaussianNormalizer
import lunzi.nn as nn


class LinearReward(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, simplex=False, sqscale=0.01,
                 favor_zero_expert_reward=False, recompute_expert_feat=False, ):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.simplex = simplex
        self.sqscale = sqscale
        self.favor_zero_expert_reward = favor_zero_expert_reward
        self.recompute_expert_feat = recompute_expert_feat

        with self.scope:
            self.normalizer = GaussianNormalizer(name="inputs", shape=[dim_state + dim_action])

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def build(self, expert_obs, expert_acs):
        self.expert_obs = expert_obs
        self.expert_acs = expert_acs
        inputs = np.concatenate([self.expert_obs, self.expert_acs], axis=1)
        self.normalizer.update(inputs)

        self.normalizer_mean, self.normalizer_std = self.normalizer.eval(fetch='mean std')
        self.normalizer_updated = False
        logger.info('mean: {}'.format(self.normalizer_mean))
        logger.info('std:{}'.format(self.normalizer_std))

        self.expert_featexp = self._compute_featexp(self.expert_obs, self.expert_acs)
        feat_dim = self.expert_featexp.shape[0]
        if self.simplex:
            self.widx = np.random.randint(feat_dim)
        else:
            self.w = np.random.randn(feat_dim)
            self.w /= np.linalg.norm(self.w) + 1e-8

        self.reward_bound = 0.
        self.gap = 0.

    def get_reward(self, states, actions):
        if len(states.shape) == 1:
            states = np.expand_dims(states, 0)
        if len(actions.shape) == 1:
            actions = np.expand_dims(actions, 0)

        feat = self._featurize(states, actions)
        rew = (feat[:, self.widx] if self.simplex else feat.dot(self.w)) / float(feat.shape[1])

        if self.favor_zero_expert_reward:
            self.reward_bound = max(self.reward_bound, rew.max())
        else:
            self.reward_bound = min(self.reward_bound, rew.min())
        rew_shifted = rew - self.reward_bound
        return rew_shifted

    def train(self, states, actions):
        curr_featexp = self._compute_featexp(states, actions)
        if self.recompute_expert_feat:
            self.expert_featexp = self._compute_featexp(self.expert_obs, self.expert_acs)

        if self.simplex:
            v = curr_featexp - self.expert_featexp
            self.widx = np.argmin(v)
            self.gap = self.expert_featexp[self.widx] - curr_featexp[self.widx]
        else:
            w = self.expert_featexp - curr_featexp
            l2 = np.linalg.norm(w)
            self.w = w / (l2 + 1e-6)
            self.gap = np.linalg.norm(self.expert_featexp - curr_featexp)

        train_info = dict(
            gap=self.gap
        )
        if self.simplex:
            train_info['w_idx'] = self.widx

        return train_info

    def _compute_featexp(self, obs, acs):
        return self._featurize(obs, acs).mean(axis=0)

    def _featurize(self, obs, acs):
        # normalize
        assert obs.ndim == 2 and acs.ndim == 2
        if self.normalizer_updated:
            mean, std = self.normalizer.eval(fetch='mean std')
        else:
            mean, std = self.normalizer_mean, self.normalizer_std
        inputs_normalized = (np.concatenate([obs, acs], axis=1) - mean) / np.maximum(std, 0.01)
        obs, acs = inputs_normalized[:, :obs.shape[1]], inputs_normalized[:, obs.shape[1]:]

        # Linear + Quadratic + Bias
        feat = [obs, acs, (self.sqscale * obs) ** 2, (self.sqscale * acs) ** 2]
        feat.append(np.ones([len(obs), 1]))
        feat = np.concatenate(feat, axis=1)

        assert feat.ndim == 2 and feat.shape[0] == obs.shape[0]
        return feat
