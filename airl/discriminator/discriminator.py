import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from trpo.utils.normalizer import Normalizers
from trpo.v_function.mlp_v_function import FCLayer
from typing import List


class Discriminator(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizers: Normalizers,
                 lr: float, gamma: float, policy_ent_coef: float, d_ent_coef=1e-3, max_grad_norm=None, disentangle_reward=False):
        super().__init__()

        self.gamma = gamma
        self.policy_ent_coef = policy_ent_coef
        self.d_ent_coef = d_ent_coef
        self.disentangle_reward = disentangle_reward

        with self.scope:
            self.op_true_states = tf.placeholder(tf.float32, [None, dim_state], "true_state")
            self.op_true_actions = tf.placeholder(tf.float32, [None, dim_action], "true_action")
            self.op_true_next_states = tf.placeholder(tf.float32, [None, dim_state], "true_next_state")
            self.op_true_log_probs = tf.placeholder(tf.float32, [None], "true_log_prob")
            self.op_fake_states = tf.placeholder(tf.float32, [None, dim_state], "fake_state")
            self.op_fake_actions = tf.placeholder(tf.float32, [None, dim_action], "fake_actions")
            self.op_fake_next_states = tf.placeholder(tf.float32, [None, dim_state], "fake_next_state")
            self.op_fake_log_probs = tf.placeholder(tf.float32, [None], "fake_log_prob")

            self.reward_net = MLPVFunction(dim_state, dim_action, hidden_sizes, normalizer=normalizers.state)
            self.value_net = MLPVFunction(dim_state, dim_action, hidden_sizes, normalizer=normalizers.state)

            self.op_loss, self.op_true_logits, self.op_fake_logits = self(
                self.op_true_states, self.op_true_actions, self.op_true_next_states, self.op_true_log_probs,
                self.op_fake_states, self.op_fake_actions, self.op_fake_next_states, self.op_fake_log_probs
            )
            # self.op_rewards = self.reward_net(self.op_fake_states)
            self.op_fake_prob = tf.nn.sigmoid(self.op_fake_logits)
            self.op_rewards = - tf.log(1 - self.op_fake_prob + 1e-6)

            optimizer = tf.train.AdamOptimizer(lr)
            params = self.reward_net.parameters() + self.value_net.parameters()
            grads_and_vars = optimizer.compute_gradients(self.op_loss, var_list=params)
            self.op_grad_norm = tf.global_norm([grad for grad, _ in grads_and_vars])
            if max_grad_norm is not None:
                clip_grads, op_grad_norm = tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], max_grad_norm)
                clip_grads_and_vars = [(grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)]
            else:
                clip_grads_and_vars = grads_and_vars
            self.op_train = optimizer.apply_gradients(clip_grads_and_vars)

    def forward(self, true_states: nn.Tensor, true_actions: nn.Tensor,
                true_next_states: nn.Tensor, true_log_probs: nn.Tensor,
                fake_states: nn.Tensor, fake_actions: nn.Tensor,
                fake_next_states: nn.Tensor, fake_log_probs: nn.Tensor):
        if self.disentangle_reward:
            true_rewards = self.reward_net(true_states, true_actions)
            true_state_values = self.value_net(true_states)
            true_next_state_values = self.value_net(true_next_states)
            true_logits = true_rewards + self.gamma * true_next_state_values - true_state_values \
                - self.policy_ent_coef * true_log_probs

            fake_rewards = self.reward_net(fake_states, fake_actions)
            fake_state_values = self.value_net(fake_states)
            fake_next_state_values = self.value_net(fake_next_states)
            fake_logits = fake_rewards + self.gamma * fake_next_state_values - fake_state_values \
                - self.policy_ent_coef * fake_log_probs

            true_loss = tf.reduce_mean(tf.nn.softplus(-true_logits))
            fake_loss = tf.reduce_mean(2 * fake_logits + tf.nn.softplus(-fake_logits))
            # fake_loss = tf.reduce_mean(tf.nn.softplus(fake_logits))

            total_loss = true_loss + fake_loss
        else:
            true_logits = self.reward_net(true_states, true_actions) - self.policy_ent_coef * true_log_probs
            fake_logits = self.reward_net(fake_states, fake_actions) - self.policy_ent_coef * fake_log_probs

            true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=true_logits, labels=tf.ones_like(true_logits)
            )
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.zeros_like(true_logits)
            )

            logits = tf.concat([true_logits, fake_logits], axis=0)
            entropy = (1. - tf.nn.sigmoid(logits)) * logits + tf.nn.softplus(-logits)
            entropy_loss = -self.d_ent_coef * tf.reduce_mean(entropy)

            total_loss = true_loss + fake_loss + entropy_loss

        return total_loss, true_logits, fake_logits

    @nn.make_method(fetch='loss')
    def get_loss(self, true_states, true_actions, true_next_states, true_log_probs,
                 fake_states, fake_actions, fake_next_states, fake_log_probs):
        pass

    @nn.make_method(fetch='rewards')
    def get_reward(self, fake_states, fake_actions, fake_log_probs): pass

    def train(self, true_states, true_actions, true_next_states, true_log_probs,
              fake_states, fake_actions, fake_next_states, fake_log_probs):
        _, loss, true_logits, fake_logits, grad_norm = \
            self.get_loss(
                true_states, true_actions, true_next_states, true_log_probs,
                fake_states, fake_actions, fake_next_states, fake_log_probs,
                fetch='train loss true_logits fake_logits grad_norm'
            )
        info = dict(
            loss=np.mean(loss),
            grad_norm=np.mean(grad_norm),
            true_logits=np.mean(true_logits),
            fake_logits=np.mean(fake_logits),
        )
        return info


class MLPVFunction(nn.Module):
    def __init__(self, dim_state, dim_action, hidden_sizes, normalizer=None):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        with self.scope:
            layers = []
            all_sizes = [dim_state + dim_action, *self.hidden_sizes]
            for i, (in_features, out_features) in enumerate(zip(all_sizes[:-1], all_sizes[1:])):
                layers.append(FCLayer(in_features, out_features))
                layers.append(nn.ReLU())
            layers.append(FCLayer(all_sizes[-1], 1))
            self.net = nn.Sequential(*layers)
            self.normalizer = normalizer
            self.op_states = tf.placeholder(tf.float32, shape=[None, dim_state])
            self.op_actions = tf.placeholder(tf.float32, shape=[None, dim_action])
            self.op_values = self.forward(self.op_states, self.op_actions)

    def forward(self, states, actions):
        inputs = tf.concat([
            self.normalizer(states),
            actions,
        ], axis=-1)
        return self.net(inputs)[:, 0]

    @nn.make_method(fetch='values')
    def get_values(self, states): pass
