import pickle
import os
import time
import random
import yaml
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.utils.normalizer import Normalizers
from sac.policies.actor import Actor
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from mbrl.gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from mbrl.gail.policies.gaussian_mlp_policy import GaussianMLPPolicy
from mbrl.gail.utils.runner import VirtualRunner, evaluate as evaluate_on_virtual_env
from gail.utils.runner import Runner, evaluate as evaluate_on_true_env
from utils.envs.mujoco.virtual_env import VirtualEnv
from utils import FLAGS, get_tf_config
from sklearn import manifold
import matplotlib.pyplot as plt


def create_env(env_id, seed, rescale_action=True):
    import gym
    from utils.envs.mujoco_wrapper import ReScaleActionWrapper

    env = gym.make('MB' + env_id)
    max_episode_steps = env._max_episode_steps
    env.seed(seed)
    env.verify()
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    if rescale_action:
        env = ReScaleActionWrapper(env)
    setattr(env, '_max_episode_steps', max_episode_steps)
    setattr(env, 'max_episode_steps', max_episode_steps)
    setattr(env, 'n_envs', 1)
    env.partial_reset = classmethod(lambda cls, indices: cls.reset())
    return env


class BehavioralCloningLoss(nn.Module):
    def __init__(self, dim_state: int, dim_action: int, policy: GaussianMLPPolicy, lr: float, train_std=False):
        super().__init__()

        with self.scope:
            self.op_states = tf.placeholder(tf.float32, [None, dim_state], "state")
            self.op_actions = tf.placeholder(tf.float32, [None, dim_action], "action")
            self.op_next_states = tf.placeholder(tf.float32, [None, dim_state], "next_state")

            distribution = policy(self.op_states, self.op_actions)
            if policy.output_diff:
                normalized_target = policy.normalizers.diff(self.op_next_states - self.op_states)
            else:
                normalized_target = policy.normalizers.state(self.op_next_states)
            if train_std:
                self.op_loss = -tf.reduce_mean(distribution.log_prob(normalized_target).reduce_sum(axis=1))
            else:
                self.op_loss = tf.reduce_mean(tf.square(distribution.mean() - normalized_target))

            optimizer = tf.train.AdamOptimizer(lr)
            grads = tf.gradients(self.op_loss, policy.parameters())
            self.op_grad_norm = tf.global_norm(grads)
            self.op_train = optimizer.minimize(self.op_loss, var_list=policy.parameters())

    def forward(self):
        raise NotImplementedError

    @nn.make_method(fetch='loss')
    def get_loss(self, states, actions, next_states): pass


def set_random_seed(seed):
    assert seed > 0 and isinstance(seed, int)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = create_env(FLAGS.env.id, seed=FLAGS.seed, rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    bc_normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    bc_policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes,
                                  output_diff=FLAGS.TRPO.output_diff, normalizers=bc_normalizers)

    gail_normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    gail_policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes,
                                    output_diff=FLAGS.TRPO.output_diff, normalizers=gail_normalizers)

    actor = Actor(dim_state, dim_action, FLAGS.SAC.actor_hidden_sizes)
    tf.get_default_session().run(tf.global_variables_initializer())

    loader = nn.ModuleDict({'actor': actor})
    policy_load = f'dataset/mb2/{FLAGS.env.id}/policy.npy'
    loader.load_state_dict(np.load(policy_load, allow_pickle=True)[()])
    logger.warning('Load expert policy from %s' % policy_load)

    bc_policy_load = "benchmarks/mbrl_benchmark/mbrl2_bc_30_1000/mbrl2_bc-Walker2d-v2-100-2020-05-22-16-02-12/final.npy"
    loader = nn.ModuleDict({'policy': bc_policy, 'normalizers': bc_normalizers})
    loader.load_state_dict(np.load(bc_policy_load, allow_pickle=True)[()])
    logger.warning('Load bc policy from %s' % bc_policy_load)

    gail_policy_load = "benchmarks/mbrl_benchmark/mbrl2_gail_grad_penalty/mbrl2_gail-Walker2d-v2-100-2020-05-22-12-10-07/final.npy"
    loader = nn.ModuleDict({'policy': gail_policy, 'normalizers': gail_normalizers})
    loader.load_state_dict(np.load(gail_policy_load, allow_pickle=True)[()])
    logger.warning('Load gail policy from %s' % gail_policy_load)

    eval_gamma = 0.999
    eval_returns, eval_lengths = evaluate_on_true_env(actor, env, gamma=eval_gamma)
    logger.warning('Test policy true value = %.4f true length = %d (gamma = %f)',
                   np.mean(eval_returns), np.mean(eval_lengths), eval_gamma)

    real_runner = Runner(env, max_steps=env.max_episode_steps, rescale_action=False)
    # virtual env
    env_bc_stochastic = VirtualEnv(bc_policy, env, n_envs=1, stochastic_model=True)
    env_bc_deterministic = VirtualEnv(bc_policy, env, n_envs=1, stochastic_model=False)
    runner_bc_stochastic = VirtualRunner(env_bc_stochastic, max_steps=env.max_episode_steps, rescale_action=False)
    runner_bc_deterministic = VirtualRunner(env_bc_deterministic, max_steps=env.max_episode_steps, rescale_action=False)

    env_gail_stochastic = VirtualEnv(gail_policy, env, n_envs=1, stochastic_model=True)
    env_gail_deterministic = VirtualEnv(gail_policy, env, n_envs=1, stochastic_model=False)
    runner_gail_stochastic = VirtualRunner(env_gail_stochastic, max_steps=env.max_episode_steps)
    runner_gail_deterministic = VirtualRunner(env_gail_deterministic, max_steps=env.max_episode_steps)

    data_actor, ep_infos = real_runner.run(actor, n_samples=int(2e3), stochastic=False)
    returns = [info['return'] for info in ep_infos]
    lengths = [info['length'] for info in ep_infos]
    logger.info('Collect %d samples for actor avg return = %.4f avg length = %d',
                len(data_actor), np.mean(returns), np.mean(lengths))

    data_bc_stochastic, ep_infos = runner_bc_stochastic.run(actor, n_samples=int(2e3), stochastic=False)
    returns = [info['return'] for info in ep_infos]
    lengths = [info['length'] for info in ep_infos]
    logger.info('Collect %d samples for bc stochastic policy avg return = %.4f avg length = %d',
                len(data_bc_stochastic), np.mean(returns), np.mean(lengths))

    reward_ref, _ = env.mb_step(data_bc_stochastic.state, data_bc_stochastic.action, data_bc_stochastic.next_state)
    np.testing.assert_allclose(reward_ref, data_bc_stochastic.reward, rtol=1e-4, atol=1e-4)

    data_bc_deterministic, ep_infos = runner_bc_deterministic.run(actor, n_samples=int(2e3), stochastic=False)
    returns = [info['return'] for info in ep_infos]
    lengths = [info['length'] for info in ep_infos]
    logger.info('Collect %d samples for bc deterministic policy avg return = %.4f avg length = %d',
                len(data_bc_deterministic), np.mean(returns), np.mean(lengths))

    reward_ref, _ = env.mb_step(data_bc_deterministic.state, data_bc_deterministic.action, data_bc_deterministic.next_state)
    np.testing.assert_allclose(reward_ref, data_bc_deterministic.reward, rtol=1e-4, atol=1e-4)

    data_gail_stochastic, ep_infos = runner_gail_stochastic.run(actor, n_samples=int(2e3), stochastic=False)
    returns = [info['return'] for info in ep_infos]
    lengths = [info['length'] for info in ep_infos]
    logger.info('Collect %d samples for gail stochastic policy avg return = %.4f avg length = %d',
                len(data_gail_stochastic), np.mean(returns), np.mean(lengths))
    data_gail_deterministic, ep_infos = runner_gail_deterministic.run(actor, n_samples=int(2e3), stochastic=False)
    returns = [info['return'] for info in ep_infos]
    lengths = [info['length'] for info in ep_infos]
    logger.info('Collect %d samples for gail deterministic policy avg return = %.4f avg length = %d',
                len(data_bc_deterministic), np.mean(returns), np.mean(lengths))

    t_sne = manifold.TSNE(init='pca', random_state=2020)
    data = np.concatenate([data.state for data in [
        data_actor, data_bc_stochastic, data_bc_deterministic,
        data_gail_stochastic, data_gail_deterministic]],
                          axis=0)
    step = np.concatenate([data.step for data in [
        data_actor, data_bc_stochastic, data_bc_deterministic,
        data_gail_stochastic, data_gail_deterministic]],
                          axis=0)
    loc, scale = bc_normalizers.state.eval('mean std')
    data = (data - loc) / (1e-6 + scale)
    embedding = t_sne.fit_transform(data)

    fig, axarrs = plt.subplots(nrows=1, ncols=5, figsize=[6*5, 4],
                               squeeze=False, sharex=True, sharey=True, dpi=300)
    start = 0
    indices = 0
    g2c = {}
    for title in ['expert', 'bc_stochastic', 'bc_deterministic', 'gail_stochastic', 'gail_deterministic']:
        g2c[title] = axarrs[0][indices].scatter(embedding[start:start+2000, 0], embedding[start:start+2000, 1],
                                   c=step[start:start+2000])
        axarrs[0][indices].set_title(title)
        indices += 1
        start += 2000
    plt.colorbar(list(g2c.values())[0], ax=axarrs.flatten())
    plt.tight_layout()
    plt.savefig(f'{FLAGS.log_dir}/visualize.png', bbox_inches='tight')

    data = {
        'expert': data_actor.state,
        'bc_stochastic': data_bc_stochastic.state,
        'bc_deterministic': data_bc_deterministic.state,
        'gail_stochastic': data_gail_stochastic.state,
        'gail_deterministic': data_gail_deterministic.state
    }
    np.savez(f'{FLAGS.log_dir}/data.npz', **data)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
