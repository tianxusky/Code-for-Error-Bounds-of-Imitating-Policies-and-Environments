import tensorflow as tf
from acer.utils.cnn_utils import FCLayer
import lunzi.nn as nn
from typing import List


class BinaryClassifier(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int],
                 state_process_fn, action_process_fn, activ_fn='none'):
        super().__init__()
        self.dim_state = dim_state
        self.dim_action = dim_action
        self.hidden_sizes = hidden_sizes
        # this avoid to save normalizer into self.state_dict
        self.state_process_fn = state_process_fn
        self.action_process_fn = action_process_fn

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state], "state")
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action], "action")

            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], 1))
            if activ_fn == 'none':
                pass
            elif activ_fn == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activ_fn == 'tanh':
                layers.append(nn.Tanh())
            else:
                raise ValueError('%s is not supported' % activ_fn)
            self.net = nn.Sequential(*layers)

    def forward(self, states: nn.Tensor, actions: nn.Tensor):
        inputs = tf.concat([
            self.state_process_fn(states), self.action_process_fn(actions)
        ], axis=-1)
        logits = self.net(inputs)[:, 0]
        return logits


