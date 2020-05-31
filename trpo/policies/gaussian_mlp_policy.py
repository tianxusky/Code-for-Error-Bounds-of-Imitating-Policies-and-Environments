# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
from lunzi import Tensor
from lunzi import nn
from acer.utils.cnn_utils import FCLayer
from trpo.utils.truncated_normal import LimitedEntNormal
from . import BasePolicy
from trpo.utils.normalizer import GaussianNormalizer


class GaussianMLPPolicy(nn.Module, BasePolicy):
    op_states: Tensor

    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizer: GaussianNormalizer,
                 init_std=1.):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.init_std = init_std
        self.normalizer = normalizer
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')
            self.op_actions_ = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')

            layers = []
            all_sizes = [dim_state, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.Tanh())
            layers.append(FCLayer(all_sizes[-1], dim_action, init_scale=0.01))
            self.net = nn.Sequential(*layers)

            self.op_log_std = nn.Parameter(
                tf.constant(np.log(self.init_std), shape=[self.dim_action], dtype=tf.float32), name='log_std')

        self.distribution = self(self.op_states)
        self.op_actions = self.distribution.sample()
        self.op_actions_mean = self.distribution.mean()
        self.op_actions_std = self.distribution.stddev()
        self.op_mse_loss = tf.reduce_mean(tf.square(self.op_actions_ - self.op_actions_mean))
        self.op_nlls_ = -self.distribution.log_prob(self.op_actions_).reduce_sum(axis=1)

    def forward(self, states):
        states = self.normalizer(states)
        actions_mean = self.net(states)
        distribution = LimitedEntNormal(actions_mean, self.op_log_std.exp())

        return distribution

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    @nn.make_method(fetch='mse_loss')
    def get_mse_loss(self, states, actions_): pass

    def clone(self):
        return GaussianMLPPolicy(self.dim_state, self.dim_action, self.hidden_sizes, self.normalizer, self.init_std)

