# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
from lunzi import Tensor
from lunzi import nn
from acer.utils.cnn_utils import FCLayer
from trpo.utils.truncated_normal import LimitedEntNormal
from . import BasePolicy
from trpo.utils.normalizer import Normalizers


class GaussianMLPPolicy(nn.Module, BasePolicy):
    op_states: Tensor

    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizers: Normalizers,
                 output_diff=False, init_std=1.):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        self.output_diff = output_diff
        self.init_std = init_std
        self.normalizers = normalizers
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')
            self.op_actions = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')
            self.op_next_states_ = tf.placeholder(tf.float32, shape=[None, dim_state], name='next_states')

            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.Tanh())
            layers.append(FCLayer(all_sizes[-1], dim_state, init_scale=0.01))
            self.net = nn.Sequential(*layers)

            self.op_log_std = nn.Parameter(
                tf.constant(np.log(self.init_std), shape=[self.dim_state], dtype=tf.float32), name='log_std')

            self.distribution = self(self.op_states, self.op_actions)
            self.op_next_states_std = self.distribution.stddev()
            if self.output_diff:
                self.op_next_states_mean = self.op_states + self.normalizers.diff(
                    self.distribution.mean(),
                    inverse=True)
                self.op_next_states = self.op_states + self.normalizers.diff(tf.clip_by_value(
                    self.distribution.sample(),
                    self.distribution.mean() - 3 * self.distribution.stddev(),
                    self.distribution.mean() + 3 * self.distribution.stddev()
                ), inverse=True)
            else:
                self.op_next_states_mean = self.normalizers.state(
                    self.distribution.mean(),
                    inverse=True)
                self.op_next_states = self.normalizers.state(tf.clip_by_value(
                    self.distribution.sample(),
                    self.distribution.mean() - 3 * self.distribution.stddev(),
                    self.distribution.mean() + 3 * self.distribution.stddev()
                ), inverse=True)
            self.op_mse_loss = tf.reduce_mean(tf.square(
                self.normalizers.state(self.op_next_states_) - self.normalizers.state(self.op_next_states_mean),
                ))

    def forward(self, states, actions):
        inputs = tf.concat([
            self.normalizers.state(states),
            actions.clip_by_value(-1., 1.)
        ], axis=1)
        normalized_outputs = self.net(inputs)

        distribution = LimitedEntNormal(normalized_outputs, self.op_log_std.exp())
        return distribution

    @nn.make_method(fetch='states')
    def get_next_states(self, states, actions): pass

    @nn.make_method(fetch='mse_loss')
    def get_mse_loss(self, states, actions, next_states_): pass

    def clone(self):
        return GaussianMLPPolicy(self.dim_state, self.dim_action, self.hidden_sizes, self.normalizers,
                                 self.output_diff, self.init_std)

