from lunzi import nn
import gym
import tensorflow as tf
import numpy as np
from acer.policies import BaseNNPolicy
from acer.utils.tf_utils import avg_norm, gradient_add, Scheduler, cat_entropy_softmax, get_by_index, q_explained_variance


class ACER(nn.Module):
    def __init__(self, state_spec: gym.spec, action_spec: gym.spec, policy: BaseNNPolicy, lr: float, lrschedule: str,
                 total_timesteps: int, ent_coef: float, q_coef: float, delta=1., alpha=0.99, c=10.0,
                 trust_region=True, max_grad_norm=10, rprop_alpha=0.99, rprop_epsilon=1e-5):
        super().__init__()
        self.state_spec = state_spec
        self.action_spec = action_spec
        self.lr = lr
        self.total_timesteps = total_timesteps
        self.q_coef = q_coef
        self.alpha = alpha
        self.delta = delta
        self.c = c
        self.ent_coef = ent_coef
        self.trust_region = trust_region
        self.max_grad_norm = max_grad_norm
        self.rprop_alpha = rprop_alpha
        self.rprop_epsilon = rprop_epsilon

        self.policy = policy
        self.old_policy = self.policy.clone()

        self.op_states = tf.placeholder(tf.float32, [None, *state_spec.shape], "states")
        self.op_actions = tf.placeholder(tf.float32, [None, *action_spec.shape], "actions")
        self.op_rewards = tf.placeholder(tf.float32, [None], "rewards")
        self.op_qrets = tf.placeholder(tf.float32, [None], "q_ret")
        self.op_mus = tf.placeholder(tf.float32, [None, action_spec.n], "mus")
        self.op_lr = tf.placeholder(tf.float32, [], "lr")
        self.op_alpha = tf.placeholder(tf.float32, [], "alpha")

        old_params, new_params = self.old_policy.parameters(), self.policy.parameters()
        self.op_update_old_policy = tf.group(
            *[tf.assign(old_v, self.op_alpha * old_v + (1 - self.op_alpha) * new_v)
              for old_v, new_v in zip(old_params, new_params)])

        self.op_loss, self.op_loss_policy, self.op_loss_f, self.op_loss_bc, self.op_loss_q, self.op_entropy, \
            self.op_grads, self.op_ev, self.op_v_values, self.op_norm_k, self.op_norm_g, self.op_norm_k_dot_g, self.op_norm_adj = \
            self.build(self.op_states, self.op_actions, self.op_mus, self.op_qrets)
        self.op_param_norm = tf.global_norm(self.policy.parameters())

        self.lr_schedule = Scheduler(v=lr, nvalues=total_timesteps, schedule=lrschedule)

        self.build_optimizer()

    @nn.make_method(fetch='update_old_policy')
    def update_old_policy(self, alpha): pass

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def build(self, states: nn.Tensor, actions: nn.Tensor, mus: nn.Tensor, qrets: nn.Tensor):
        c, delta, eps, q_coef, ent_coef = self.c, self.delta, 1e-6, self.q_coef, self.ent_coef
        # build v-function
        pi_logits, q = self.policy(states)
        f = tf.nn.softmax(pi_logits)
        f_pol = tf.nn.softmax(self.old_policy(states)[0])
        v = tf.reduce_sum(f * q, axis=-1)

        f_i = get_by_index(f, actions)
        q_i = get_by_index(q, actions)
        rho = f / (mus + eps)
        rho_i = get_by_index(rho, actions)

        # Calculate losses
        # Entropy
        entropy = cat_entropy_softmax(f)

        # Truncated importance sampling
        adv = qrets - v
        logf = tf.log(f_i + eps)
        gain_f = logf * tf.stop_gradient(adv * tf.minimum(c, rho_i))  # [nenvs * nsteps]
        loss_f = -gain_f
        # Bias correction for the truncation
        adv_bc = q - tf.reshape(v, (-1, 1))
        logf_bc = tf.log(f + eps)
        # IMP: This is sum, as expectation wrt f
        gain_bc = tf.reduce_sum(logf_bc * tf.stop_gradient(adv_bc * tf.nn.relu(1.0 - (c / (rho + eps))) * f), axis=1)
        loss_bc = -gain_bc

        loss_policy = loss_f + loss_bc

        loss_q = tf.square(tf.stop_gradient(qrets) - q_i)*0.5
        ev = q_explained_variance(q_i, qrets)
        # Net loss
        loss = tf.reduce_mean(loss_policy) + q_coef * tf.reduce_mean(loss_q) - ent_coef * tf.reduce_mean(entropy)

        params = self.policy.parameters()

        if self.trust_region:
            g = tf.gradients(-(loss_policy - ent_coef * entropy), f)  # [nenvs * nsteps, nact]
            # k = tf.gradients(KL(f_pol || f), f)
            k = - f_pol / (f + eps)  # [nenvs * nsteps, nact] # Directly computed gradient of KL divergence wrt f
            k_dot_g = tf.reduce_sum(k * g, axis=-1)
            adj = tf.maximum(0.0, (tf.reduce_sum(k * g, axis=-1) - delta) /
                             (tf.reduce_sum(tf.square(k), axis=-1) + eps))  # [nenvs * nsteps]

            # Calculate stats (before doing adjustment) for logging.
            avg_norm_k = avg_norm(k)
            avg_norm_g = avg_norm(g)
            avg_norm_k_dot_g = tf.reduce_mean(tf.abs(k_dot_g))
            avg_norm_adj = tf.reduce_mean(tf.abs(adj))

            g = (g - tf.reshape(adj, [-1, 1]) * k)
            sh = g.get_shape().as_list()
            assert len(sh) == 3 and sh[0] == 1
            g = g[0]
            grads_f = -g / tf.cast(tf.shape(g)[0], tf.float32)  # These are turst region adjusted gradients wrt f ie statistics of policy pi
            grads_policy = tf.gradients(f, params, grads_f)
            grads_q = tf.gradients(tf.reduce_mean(loss_q) * q_coef, params)
            grads = [gradient_add(g1, g2, param) for (g1, g2, param) in zip(grads_policy, grads_q, params)]
        else:
            grads = tf.gradients(loss, params)
            avg_norm_k, avg_norm_g, avg_norm_k_dot_g, avg_norm_adj = tf.zeros([]), tf.zeros([]), tf.zeros([]), tf.zeros([])

        return loss, tf.reduce_mean(loss_policy), tf.reduce_mean(loss_f), tf.reduce_mean(loss_bc),\
            tf.reduce_mean(loss_q), tf.reduce_mean(entropy), grads, ev, tf.reduce_mean(v), \
            avg_norm_k, avg_norm_g, avg_norm_k_dot_g, avg_norm_adj

    def build_optimizer(self):
        self.op_grad_norm = tf.global_norm(self.op_grads)
        if self.max_grad_norm is not None:
            grads, _ = tf.clip_by_global_norm(self.op_grads, self.max_grad_norm, self.op_grad_norm)
        else:
            grads = self.op_grads
        params = self.policy.parameters()
        grads = list(zip(grads, params))
        trainer = tf.train.RMSPropOptimizer(learning_rate=self.op_lr, decay=self.rprop_alpha, epsilon=self.rprop_epsilon)
        self.op_train = trainer.apply_gradients(grads)

    @nn.make_method(fetch='train')
    def optimize(self, states, actions, qrets, mus, lr): pass

    def train(self, data, qret: np.ndarray, current_steps: int):
        lr = self.lr_schedule.value_steps(current_steps)
        _, loss_policy, loss_bc, loss_q, entropy, grad_norm, param_norm, ev, v_values,\
            norm_k, norm_g, norm_adj, k_dot_g = self.optimize(
                data.state, data.action, qret, data.mu, lr,
                fetch='train loss_f loss_bc loss_q entropy grad_norm param_norm ev v_values '
                      'norm_k norm_g norm_adj norm_k_dot_g')
        self.update_old_policy(self.alpha)

        for param in self.parameters():
            param.invalidate()

        info = dict(
            loss_policy=loss_policy,
            loss_bc=loss_bc,
            loss_q=loss_q,
            entropy=entropy,
            grad_norm=grad_norm,
            param_norm=param_norm,
            ev=ev,
            v_values=v_values,
            norm_k=norm_k,
            norm_g=norm_g,
            norm_adj=norm_adj,
            k_dot_g=k_dot_g
        )

        return info


