import tensorflow as tf
from acer.utils.cnn_utils import FCLayer
import lunzi.nn as nn


class Critic(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_sizes):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[None, dim_action])
            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], 1))
            self.net1 = nn.Sequential(*layers)
            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], 1))
            self.net2 = nn.Sequential(*layers)

        self.op_q1, self.op_q2 = self.forward(self.op_states, self.op_actions)

    def forward(self, states: nn.Tensor, actions: nn.Tensor):
        x = tf.concat([states, actions], axis=-1)
        q1 = self.net1(x)[:, 0]
        q2 = self.net2(x)[:, 0]
        return q1, q2

    def clone(self):
        return Critic(self.dim_state, self.dim_action, self.hidden_sizes)

    @nn.make_method(fetch='q1 q2')
    def get_q_values(self, states, actions): pass

