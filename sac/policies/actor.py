# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from typing import List
import tensorflow as tf
import numpy as np
from lunzi import Tensor
from lunzi import nn
from acer.utils.cnn_utils import FCLayer
import tensorflow_probability as tfp

ds = tfp.distributions

LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPS = 1e-6


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


class Actor(nn.Module):
    op_states: Tensor

    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int]):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state], name='states')
            self.op_actions_ = tf.placeholder(tf.float32, shape=[None, dim_action], name='actions')

            layers = []
            all_sizes = [dim_state, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], dim_action*2))
            self.net = nn.Sequential(*layers)

        self.op_actions, self.op_log_density, pd, self.op_dist_mean, self.op_dist_log_std = self(self.op_states)
        self.op_actions_mean = tf.tanh(self.op_dist_mean)
        pi_ = tf.atanh(clip_but_pass_gradient(self.op_actions_, -1+EPS, 1-EPS))
        log_prob_pi_ = pd.log_prob(pi_).reduce_sum(axis=1)
        log_prob_pi_ -= tf.reduce_sum(tf.log(1 - self.op_actions_ ** 2 + EPS), axis=1)
        self.op_log_density_ = log_prob_pi_

    def forward(self, states):
        out = self.net(states)
        mu, log_std = tf.split(out, num_or_size_splits=2, axis=1)
        log_std = tf.nn.tanh(log_std)
        assert LOG_STD_MAX > LOG_STD_MIN
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        std = tf.exp(log_std)

        pd = tf.distributions.Normal(mu, std)
        pi = pd.sample()
        log_prob_pi = pd.log_prob(pi).reduce_sum(axis=1)
        log_prob_pi -= tf.reduce_sum(tf.log(1 - tf.tanh(pi) ** 2 + EPS), axis=1)
        actions = tf.tanh(pi)
        return actions, log_prob_pi, pd, mu, log_std

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    @nn.make_method(fetch='log_density_')
    def get_log_density(self, states, actions_): pass

    def clone(self):
        return Actor(self.dim_state, self.dim_action, self.hidden_sizes)


if __name__ == '__main__':
    with tf.Session() as sess:
        actor = Actor(10, 3, [256, 256])
        sess.run(tf.global_variables_initializer())

        states__ = np.random.randn(2000, 10)
        actions__, log_density = actor.get_actions(states__, fetch='actions log_density')
        log_density_ref = actor.get_log_density(states__, actions__)

        np.testing.assert_allclose(log_density, log_density_ref, rtol=5e-4)
