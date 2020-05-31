from lunzi import nn
import tensorflow as tf
import numpy as np
from acer.utils.cnn_utils import FCLayer
from acer.utils.distributions import CategoricalPd
from acer.utils.tf_utils import get_by_index


class MLPPolicy(nn.Module):
    def __init__(self, state_spec, action_spec, hidden_sizes=(64, 64)):
        super().__init__()
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.hidden_sizes = hidden_sizes

        self.op_states = tf.placeholder(state_spec.dtype, [None, *state_spec.shape], 'states')
        self.op_actions_ = tf.placeholder(action_spec.dtype, [None, *action_spec.shape], 'actions')

        all_sizes = [state_spec.shape[0], *hidden_sizes]
        layer = []
        for nin, nh in zip(all_sizes[:-1], all_sizes[1:]):
            layer.append(FCLayer(nin, nh, init_scale=np.sqrt(2)))
            layer.append(nn.Tanh())
        self.mlp_net = nn.Sequential(*layer)
        self.pi_net = FCLayer(all_sizes[-1], action_spec.n, init_scale=0.01)
        self.q_net = FCLayer(all_sizes[-1], action_spec.n)

        pi_logits, q_values, = self.forward(self.op_states)
        self.pd = CategoricalPd(pi_logits)
        self.op_actions = self.pd.sample()
        self.op_actions_mean = self.pd.mode()
        self.op_mus = tf.nn.softmax(pi_logits)
        self.op_v_values = tf.reduce_sum(self.op_mus * q_values, axis=-1)
        self.op_nlls = self.pd.neglogp(self.op_actions)
        self.op_q_values = get_by_index(q_values, self.op_actions)
        self.op_q_values_ = get_by_index(q_values, self.op_actions_)

    def forward(self, states):
        h = self.mlp_net(states)
        pi_logits = self.pi_net(h)
        q_values = self.q_net(h)
        return pi_logits, q_values

    @nn.make_method(fetch='actions')
    def get_actions(self, states): pass

    @nn.make_method(fetch='q_values_')
    def get_q_values(self, states, actions_): pass

    @nn.make_method(fetch='v_values')
    def get_v_values(self, states): pass

    @nn.make_method(fetch='mus')
    def get_mus(self, states): pass

    def clone(self):
        return MLPPolicy(self.state_spec, self.action_spec, self.hidden_sizes)

