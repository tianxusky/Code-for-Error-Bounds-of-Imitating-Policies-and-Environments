import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger
from trpo.utils.normalizer import Normalizers
from .binary_classifier import BinaryClassifier
from typing import List


class Discriminator(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, hidden_sizes: List[int], normalizers: Normalizers,
                 lr: float, ent_coef: float, loc=None, scale=None,
                 neural_distance=False, gradient_penalty_coef=0., l2_regularization_coef=0.,
                 max_grad_norm=None, subsampling_rate=20.):
        super().__init__()
        self.ent_coef = ent_coef
        self.neural_distance = neural_distance
        self.gradient_penalty_coef = gradient_penalty_coef
        self.l2_regularization_coef = l2_regularization_coef
        self.subsampling_rate = subsampling_rate

        with self.scope:
            self.op_true_states = tf.placeholder(tf.float32, [None, dim_state], "true_state")
            self.op_true_actions = tf.placeholder(tf.float32, [None, dim_action], "true_action")
            self.op_fake_states = tf.placeholder(tf.float32, [None, dim_state], "fake_state")
            self.op_fake_actions = tf.placeholder(tf.float32, [None, dim_action], "fake_actions")
            self.op_true_masks = tf.placeholder(tf.float32, [None], "mask")

            if self.neural_distance or self.gradient_penalty_coef > 0.:
                logger.info('Use predefined normalization.')
                if loc is None:   loc = np.zeros([1, dim_state], dtype=np.float32)
                if scale is None: scale = np.ones_like([1, dim_action], dtype=np.float32)
                logger.info('Normalizer loc:{} \n scale:{}'.format(loc, scale))
                state_process_fn = lambda states_: (states_ - loc) / (1e-3 + scale)
            else:
                logger.info('Use given normalizer.')
                state_process_fn = lambda states_: normalizers.state(states_)
            action_process_fn = lambda action_: action_
            activ_fn = 'none'
            if self.neural_distance:
                activ_fn = 'none'

            self.classifier = BinaryClassifier(dim_state, dim_action, hidden_sizes,
                                               state_process_fn=state_process_fn,
                                               action_process_fn=action_process_fn,
                                               activ_fn=activ_fn)

            self.op_loss, self.op_classifier_loss, self.op_entropy_loss, self.op_grad_penalty, self.op_regularization, \
                self.op_true_logits, self.op_fake_logits, self.op_true_weight = self(
                    self.op_true_states, self.op_true_actions,
                    self.op_fake_states, self.op_fake_actions,
                    self.op_true_masks)
            self.op_true_prob = tf.nn.sigmoid(self.op_true_logits)
            self.op_fake_prob = tf.nn.sigmoid(self.op_fake_logits)

            optimizer = tf.train.AdamOptimizer(lr)
            params = self.classifier.parameters()
            grads_and_vars = optimizer.compute_gradients(self.op_loss, var_list=params)
            if max_grad_norm is not None:
                clip_grads, op_grad_norm = tf.clip_by_global_norm([grad for grad, _ in grads_and_vars], max_grad_norm)
                clip_grads_and_vars = [(grad, var) for grad, (_, var) in zip(clip_grads, grads_and_vars)]
            else:
                op_grad_norm = tf.global_norm([grad for grad, _ in grads_and_vars])
                clip_grads_and_vars = grads_and_vars
            self.op_train = optimizer.apply_gradients(clip_grads_and_vars)
            if self.neural_distance:
                logger.info('Discriminator uses Wasserstein distance.')
            logger.info('{}'.format(self.classifier.parameters()))
            logger.info('Use gradient penalty regularization (coef = %f)', gradient_penalty_coef)
            self.op_grad_norm = op_grad_norm
            # neural reward function
            reference = tf.reduce_mean(self.op_fake_logits)
            self.op_unscaled_neural_reward = self.op_fake_logits
            unscaled_reward = self.op_fake_logits - reference
            reward_scale = tf.reduce_max(unscaled_reward) - tf.reduce_min(unscaled_reward)
            self.op_scaled_neural_reward = unscaled_reward / (1e-6 + reward_scale)
            # gail reward function
            self.op_gail_reward = - tf.log(1 - self.op_fake_prob + 1e-6)

    def forward(self, true_states: nn.Tensor, true_actions: nn.Tensor, fake_states: nn.Tensor, fake_actions: nn.Tensor,
                true_masks: nn.Tensor):
        true_logits = self.classifier(true_states, true_actions)
        fake_logits = self.classifier(fake_states, fake_actions)

        true_masks = tf.maximum(0., -true_masks)
        true_weight = true_masks / self.subsampling_rate + (1 - true_masks)

        if self.neural_distance:
            classify_loss = tf.reduce_mean(fake_logits) - tf.reduce_mean(true_logits * true_weight)
            entropy_loss = tf.zeros([])
        else:
            true_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=true_logits, labels=tf.ones_like(true_logits)
            )
            fake_loss = tf.nn.sigmoid_cross_entropy_with_logits(
                logits=fake_logits, labels=tf.zeros_like(fake_logits)
            )

            classify_loss = tf.reduce_mean(true_loss * true_weight) + tf.reduce_mean(fake_loss)
            logits = tf.concat([true_logits, fake_logits], axis=0)
            entropy = (1. - tf.nn.sigmoid(logits)) * logits + tf.nn.softplus(-logits)
            entropy_loss = -self.ent_coef * tf.reduce_mean(entropy)

        alpha = tf.random_uniform(shape=[tf.shape(true_logits)[0], 1])
        inter_states = alpha * fake_states + (1 - alpha) * true_states
        inter_actions = alpha * fake_actions + (1 - alpha) * true_actions
        grad = tf.gradients(self.classifier(inter_states, inter_actions), [inter_states, inter_actions])
        grad = tf.concat(grad, axis=1)
        grad_penalty = self.gradient_penalty_coef * tf.reduce_mean(tf.pow(tf.norm(grad, axis=-1) - 1, 2))

        regularization = self.l2_regularization_coef * tf.add_n(
            [tf.nn.l2_loss(t) for t in self.classifier.parameters()],
            name='regularization')

        loss = classify_loss + entropy_loss + grad_penalty + regularization
        return loss, classify_loss, entropy_loss, grad_penalty, regularization, true_logits, fake_logits, true_weight

    @nn.make_method(fetch='loss')
    def get_loss(self, true_states, true_actions, fake_states, fake_actions, true_masks):
        pass

    @nn.make_method(fetch='unscaled_neural_reward')
    def get_neural_network_reward(self, fake_states, fake_actions):
        pass

    @nn.make_method(fetch='gail_reward')
    def get_gail_reward(self, fake_states, fake_actions):
        pass

    def get_reward(self, states, actions):
        if not self.neural_distance:
            return self.get_gail_reward(states, actions)
        else:
            return self.get_neural_network_reward(states, actions, fetch='scaled_neural_reward')

    def train(self, true_states, true_actions, fake_states, fake_actions, true_masks=None):
        if true_masks is None:
            true_masks = np.zeros([len(true_states), ], dtype=np.float32)
        _, loss, true_logits, fake_logits, true_prob, fake_prob, grad_norm, grad_penalty, regularization = \
            self.get_loss(
                true_states, true_actions, fake_states, fake_actions, true_masks,
                fetch='train loss true_logits fake_logits true_prob fake_prob grad_norm grad_penalty regularization'
            )
        info = dict(
            loss=np.mean(loss),
            grad_norm=np.mean(grad_norm),
            grad_penalty=np.mean(grad_penalty),
            regularization=np.mean(regularization),
            true_logits=np.mean(true_logits),
            fake_logits=np.mean(fake_logits),
            true_prob=np.mean(true_prob),
            fake_prob=np.mean(fake_prob),
        )
        return info
