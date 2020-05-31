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

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes,
                               output_diff=FLAGS.TRPO.output_diff, normalizers=normalizers)
    bc_loss = BehavioralCloningLoss(dim_state, dim_action, policy, lr=float(FLAGS.BC.lr), train_std=FLAGS.BC.train_std)

    actor = Actor(dim_state, dim_action, FLAGS.SAC.actor_hidden_sizes)
    tf.get_default_session().run(tf.global_variables_initializer())

    subsampling_rate = env.max_episode_steps // FLAGS.GAIL.trajectory_size
    # load expert dataset
    set_random_seed(2020)
    expert_dataset = load_expert_dataset(FLAGS.GAIL.buf_load)
    expert_state = np.stack([t.obs for t in expert_dataset.buffer()])
    expert_next_state = np.stack([t.next_obs for t in expert_dataset.buffer()])
    expert_done = np.stack([t.done for t in expert_dataset.buffer()])
    np.testing.assert_allclose(expert_next_state[:-1]*(1-expert_done[:-1][:, None]),
                               expert_state[1:]*(1-expert_done[:-1][:, None]))
    del expert_state, expert_next_state, expert_done
    expert_reward = expert_dataset.get_average_reward()
    logger.info('Expert Reward %f', expert_reward)
    if FLAGS.GAIL.learn_absorbing:
        expert_dataset.add_absorbing_states(env)
    eval_batch = expert_dataset.sample(1024)
    eval_state = np.stack([t.obs for t in eval_batch])
    eval_action = np.stack([t.action for t in eval_batch])
    eval_next_state = np.stack([t.next_obs for t in eval_batch])
    logger.info('Sampled obs: %.4f, acs: %.4f', np.mean(eval_state), np.mean(eval_action))
    expert_dataset.subsample_trajectories(FLAGS.GAIL.traj_limit)
    logger.info('Original dataset size {}'.format(len(expert_dataset)))
    expert_dataset.subsample_transitions(subsampling_rate)
    logger.info('Subsampled dataset size {}'.format(len(expert_dataset)))
    logger.info('np random: %d random : %d', np.random.randint(1000), random.randint(0, 1000))
    set_random_seed(FLAGS.seed)

    loader = nn.ModuleDict({'actor': actor})
    loader.load_state_dict(np.load(FLAGS.ckpt.policy_load, allow_pickle=True)[()])
    logger.warning('Load expert policy from %s' % FLAGS.ckpt.policy_load)
    saver = nn.ModuleDict({'policy': policy, 'normalizers': normalizers})
    print(saver)

    # updater normalizer
    expert_state = np.stack([t.obs for t in expert_dataset.buffer()])
    expert_action = np.stack([t.action for t in expert_dataset.buffer()])
    expert_next_state = np.stack([t.next_obs for t in expert_dataset.buffer()])
    normalizers.state.update(expert_state)
    normalizers.action.update(expert_action)
    normalizers.diff.update(expert_next_state - expert_state)
    del expert_state, expert_action, expert_next_state

    eval_gamma = 0.999
    eval_returns, eval_lengths = evaluate_on_true_env(actor, env, gamma=eval_gamma)
    logger.warning('Test policy true value = %.4f true length = %d (gamma = %f)',
                   np.mean(eval_returns), np.mean(eval_lengths), eval_gamma)

    # virtual env
    env_eval_stochastic = VirtualEnv(policy, env, n_envs=4, stochastic_model=True)
    env_eval_deterministic = VirtualEnv(policy, env, n_envs=4, stochastic_model=False)

    batch_size = FLAGS.BC.batch_size
    true_return = np.mean(eval_returns)
    for t in range(FLAGS.BC.max_iters):
        if t % FLAGS.BC.eval_freq == 0:
            eval_returns_stochastic, eval_lengths_stochastic = evaluate_on_virtual_env(
                actor, env_eval_stochastic, gamma=eval_gamma)
            eval_returns_deterministic, eval_lengths_deterministic = evaluate_on_virtual_env(
                actor, env_eval_deterministic, gamma=eval_gamma)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, stochastic_episode=dict(
                    returns=np.mean(eval_returns_stochastic), lengths=int(np.mean(eval_lengths_stochastic))
                ), episode=dict(
                    returns=np.mean(eval_returns_deterministic), lengths=int(np.mean(eval_lengths_deterministic))
                ),  evaluation_error=dict(
                    stochastic_error=true_return-np.mean(eval_returns_stochastic),
                    stochastic_abs=np.abs(true_return-np.mean(eval_returns_stochastic)),
                    stochastic_rel=np.abs(true_return-np.mean(eval_returns_stochastic))/true_return,
                    deterministic_error=true_return-np.mean(eval_returns_deterministic),
                    deterministic_abs=np.abs(true_return - np.mean(eval_returns_deterministic)),
                    deterministic_rel=np.abs(true_return-np.mean(eval_returns_deterministic))/true_return
                )
            ))

        expert_batch = expert_dataset.sample(batch_size)
        expert_state = np.stack([t.obs for t in expert_batch])
        expert_action = np.stack([t.action for t in expert_batch])
        expert_next_state = np.stack([t.next_obs for t in expert_batch])
        _, loss, grad_norm = bc_loss.get_loss(expert_state, expert_action, expert_next_state,
                                              fetch='train loss grad_norm')

        if t % 100 == 0:
            train_mse_loss = policy.get_mse_loss(expert_state, expert_action, expert_next_state)
            eval_mse_loss = policy.get_mse_loss(eval_state, eval_action, eval_next_state)
            log_kvs(prefix='BC', kvs=dict(
                iter=t, grad_norm=grad_norm, loss=loss, mse_loss=dict(train=train_mse_loss, eval=eval_mse_loss)
            ))

    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())

    dict_result = dict()
    for gamma in [0.9, 0.99, 0.999, 1.0]:
        eval_returns, eval_lengths = evaluate_on_virtual_env(actor, env_eval_stochastic, gamma=gamma)
        dict_result[gamma] = [float(np.mean(eval_returns)), eval_returns]
        logger.info('[%s]: %.4f', gamma, np.mean(eval_returns))

    save_path = os.path.join(FLAGS.log_dir, 'evaluate.yml')
    yaml.dump(dict_result, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
