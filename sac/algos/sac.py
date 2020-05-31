import tensorflow as tf
import numpy as np
from lunzi import nn
from sac.policies.actor import Actor
from sac.policies.critic import Critic

EPS = 1e-6


class SAC(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, actor: Actor, critic: Critic, init_alpha: float,
                 gamma: float, target_entropy: float, actor_lr: float, critic_lr: float, alpha_lr: float,
                 tau: float, actor_update_freq: int, target_update_freq: int, learn_alpha: bool):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.critic_target = self.critic.clone()
        self.gamma = gamma
        self.target_entropy = target_entropy
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.alpha_lr = alpha_lr
        self.tau = tau
        self.actor_update_freq = actor_update_freq
        self.target_update_freq = target_update_freq
        self.learn_alpha = learn_alpha

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state], 'states')
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action], 'actions')
            self.op_next_states = tf.placeholder(tf.float32, [None, dim_state], 'next_states')
            self.op_rewards = tf.placeholder(tf.float32, [None], 'rewards')
            self.op_terminals = tf.placeholder(tf.float32, [None], 'terminals')
            self.op_tau = tf.placeholder(tf.float32, [], 'tau')

            self.op_log_alpha = nn.Parameter(tf.log(init_alpha), name="log_alpha")

            target_params, source_params = self.critic_target.parameters(), self.critic.parameters()
            self.op_update_critic_target = tf.group(
                *[tf.assign(v_t, self.op_tau * v_t + (1 - self.op_tau) * v_s)
                  for v_t, v_s in zip(target_params, source_params)])

            self.op_actor_loss, self.op_critic_loss, self.op_alpha_loss, self.op_entropy, self.op_q_value, \
                self.op_dist_mean, self.op_dist_std, self.op_a1, self.op_a2, self.op_log_prob_a1 = self(
                        states=self.op_states, actions=self.op_actions, next_states=self.op_next_states,
                        rewards=self.op_rewards, terminals=self.op_terminals, log_alpha=self.op_log_alpha
                    )

            actor_optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)
            critic_optimizer = tf.train.AdamOptimizer(learning_rate=self.critic_lr)
            alpha_optimizer = tf.train.AdamOptimizer(learning_rate=self.alpha_lr)

            self.op_actor_train = actor_optimizer.minimize(self.op_actor_loss, var_list=self.actor.parameters())
            self.op_critic_train = critic_optimizer.minimize(self.op_critic_loss, var_list=self.critic.parameters())
            self.op_alpha_train = alpha_optimizer.minimize(self.op_alpha_loss, var_list=[self.op_log_alpha])

            self.op_actor_norm = tf.global_norm(self.actor.parameters())
            self.op_critic_norm = tf.global_norm(self.critic.parameters())

            self.op_alpha = tf.exp(self.op_log_alpha)
        self.iterations = 0

    def forward(self, states: nn.Tensor, actions: nn.Tensor, next_states: nn.Tensor, rewards: nn.Tensor,
                terminals: nn.Tensor, log_alpha: nn.Parameter):
        # actor
        a1, log_prob_a1, _, dist_mean, dist_log_std = self.actor(states)
        q1, q2 = self.critic(states, a1)
        q = tf.minimum(q1, q2)
        actor_loss = tf.reduce_mean(tf.exp(log_alpha) * log_prob_a1 - q)
        entropy = -tf.reduce_mean(log_prob_a1)

        # critic
        a2, log_prob_a2, *_ = self.actor(next_states)
        q1_target, q2_target = self.critic_target(next_states, a2)
        q1_predict, q2_predict = self.critic(states, actions)

        v_target = tf.minimum(q1_target, q2_target) - tf.exp(log_alpha) * log_prob_a2
        q_target = tf.stop_gradient(rewards + self.gamma * (1-terminals) * v_target)
        critic_loss = tf.reduce_mean(tf.square(q1_predict - q_target)) + tf.reduce_mean(tf.square(q2_predict - q_target))
        q_value = tf.reduce_mean(q_target)

        # alpha
        alpha = tf.exp(log_alpha)
        alpha_loss = tf.reduce_mean(alpha * (-log_prob_a1 - self.target_entropy))

        dist_mean = tf.reduce_mean(tf.tanh(dist_mean))
        dist_std = tf.reduce_mean(tf.exp(dist_log_std))
        return actor_loss, critic_loss, alpha_loss, entropy, q_value, dist_mean, dist_std, a1, a2, log_prob_a1

    @nn.make_method(fetch='critic_train critic_loss')
    def optimize_critic(self, states, actions, next_states, rewards, terminals): pass

    @nn.make_method(fetch='actor_train actor_loss')
    def optimize_actor(self, states, actions): pass

    @nn.make_method(fetch='alpha_train alpha_loss')
    def optimize_alpha(self, states): pass

    @nn.make_method(fetch='update_critic_target')
    def update_critic_target(self, tau): pass

    def train(self, data):
        _, critic_loss, q_value, critic_norm = self.optimize_critic(
            states=data.state, actions=data.action, next_states=data.next_state, rewards=data.reward,
            terminals=data.done,
            fetch='critic_train critic_loss q_value critic_norm'
        )
        assert np.isfinite(critic_loss), 'critic_loss is Nan'

        for param in self.critic.parameters():
            param.invalidate()

        if self.iterations % self.actor_update_freq == 0:
            _, actor_loss, entropy, actor_norm, dist_mean, dist_std = self.optimize_actor(
                states=data.state, actions=data.action,
                fetch='actor_train actor_loss entropy actor_norm dist_mean dist_std'
            )
            if self.learn_alpha:
                _, alpha_loss, alpha = self.optimize_alpha(
                    states=data.state,
                    fetch='alpha_train alpha_loss alpha'
                )
            else:
                alpha_loss, alpha = self.optimize_alpha(
                    states=data.state,
                    fetch='alpha_loss alpha'
                )
        else:
            actor_loss, entropy, actor_norm, dist_mean, dist_std = self.optimize_actor(
                states=data.state, actions=data.action,
                fetch='actor_loss entropy actor_norm dist_mean dist_std'
            )
            alpha_loss, alpha = self.optimize_alpha(
                states=data.state,
                fetch='alpha_loss alpha'
            )

        assert np.isfinite(actor_loss), 'actor_loss is Nan, entropy:{}'.format(entropy)
        for param in self.actor.parameters():
            param.invalidate()

        assert np.isfinite(alpha_loss), 'alpha_loss is Nan'
        assert np.isfinite(alpha), 'alpha is Nan'
        if self.iterations % self.target_update_freq == 0:
            self.update_critic_target(tau=self.tau)
        self.iterations += 1

        info = dict(
            actor_loss=actor_loss,
            critic_loss=critic_loss,
            alpha_loss=alpha_loss,
            alpha=alpha,
            entropy=entropy,
            q_value=q_value,
            dist_mean=dist_mean,
            dist_std=dist_std
        )
        return info



