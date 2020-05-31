# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import tensorflow as tf
from acer.utils.cnn_utils import FCLayer
import lunzi.nn as nn
from . import BaseVFunction


class MLPVFunction(BaseVFunction, nn.Module):
    def __init__(self, dim_state, hidden_sizes, normalizer=None):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        layers = []
        all_sizes = [dim_state, *self.hidden_sizes]
        for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
            layers.append(FCLayer(in_features, out_features))
            layers.append(nn.Tanh())
        layers.append(FCLayer(all_sizes[-1], 1))
        self.net = nn.Sequential(*layers)
        self.normalizer = normalizer
        self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state])
        self.op_values = self.forward(self.op_states)

    def forward(self, states):
        states = self.normalizer(states)
        return self.net(states)[:, 0]

    @nn.make_method(fetch='values')
    def get_values(self, states): pass

