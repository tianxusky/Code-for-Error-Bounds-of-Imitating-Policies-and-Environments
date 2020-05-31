import pickle
import os
import time
import random
import yaml
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.utils.normalizer import Normalizers
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from gail.utils.runner import Runner, evaluate
from sac.policies.actor import Actor
from utils import FLAGS, get_tf_config


"""Please Download Dataset from (https://github.com/ikostrikov/gail-experts).
Then run the following cmd to convert the dataset from h5py into a TensorFlow object.
   python -m gail.utils.replay_buffer
"""


def create_env(env_id, seed, log_dir, absorbing_state=True, rescale_action=True):
    import gym
    from utils.envs.mujoco_wrapper import ReScaleActionWrapper, AbsorbingWrapper
    from utils.envs.monitor import Monitor

    env = gym.make(env_id)
    max_episode_steps = env._max_episode_steps
    env.seed(seed)
    try:
        env.action_space.seed(seed)
    except AttributeError:
        pass
    env = Monitor(env, os.path.join(log_dir, '%d' % seed), allow_early_resets=True)
    if rescale_action:
        env = ReScaleActionWrapper(env)
    if absorbing_state:
        env = AbsorbingWrapper(env)
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

            distribution = policy(self.op_states)
            if train_std:
                self.op_loss = -tf.reduce_mean(distribution.log_prob(self.op_actions).reduce_sum(axis=1))
            else:
                self.op_loss = tf.reduce_mean(tf.square(distribution.mean() - self.op_actions))

            optimizer = tf.train.AdamOptimizer(lr)
            grads = tf.gradients(self.op_loss, policy.parameters())
            self.op_grad_norm = tf.global_norm(grads)
            self.op_train = optimizer.minimize(self.op_loss, var_list=policy.parameters())

    def forward(self):
        raise NotImplementedError

    @nn.make_method(fetch='loss')
    def get_loss(self, states, actions): pass


def set_random_seed(seed):
    assert seed > 0 and isinstance(seed, int)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = create_env(FLAGS.env.id, seed=FLAGS.seed, log_dir=FLAGS.log_dir, absorbing_state=FLAGS.GAIL.learn_absorbing,
                     rescale_action=FLAGS.env.rescale_action)
    env_eval = create_env(FLAGS.env.id, seed=FLAGS.seed+1000, log_dir=FLAGS.log_dir, absorbing_state=FLAGS.GAIL.learn_absorbing,
                          rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    bc_loss = BehavioralCloningLoss(dim_state, dim_action, policy, lr=float(FLAGS.BC.lr), train_std=FLAGS.BC.train_std)

    expert_actor = Actor(dim_state, dim_action, FLAGS.SAC.actor_hidden_sizes)
    tf.get_default_session().run(tf.global_variables_initializer())

    loader = nn.ModuleDict({'actor': expert_actor})
    if FLAGS.BC.dagger:
        loader.load_state_dict(np.load(FLAGS.ckpt.policy_load, allow_pickle=True)[()])
        logger.warning('Load expert policy from %s' % FLAGS.ckpt.policy_load)
    runner = Runner(env, max_steps=env.max_episode_steps, rescale_action=False)

    subsampling_rate = env.max_episode_steps // FLAGS.GAIL.trajectory_size
    # load expert dataset
    set_random_seed(2020)
    expert_dataset = load_expert_dataset(FLAGS.GAIL.buf_load)
    expert_reward = expert_dataset.get_average_reward()
    logger.info('Expert Reward %f', expert_reward)
    if FLAGS.GAIL.learn_absorbing:
        expert_dataset.add_absorbing_states(env)
    expert_dataset.subsample_trajectories(FLAGS.GAIL.traj_limit)
    logger.info('Original dataset size {}'.format(len(expert_dataset)))
    expert_dataset.subsample_transitions(subsampling_rate)
    logger.info('Subsampled dataset size {}'.format(len(expert_dataset)))
    logger.info('np random: %d random : %d', np.random.randint(1000), random.randint(0, 1000))
    expert_batch = expert_dataset.sample(10)
    expert_state = np.stack([t.obs for t in expert_batch])
    expert_action = np.stack([t.action for t in expert_batch])
    logger.info('Sampled obs: %.4f, acs: %.4f', np.mean(expert_state), np.mean(expert_action))
    del expert_batch, expert_state, expert_action
    set_random_seed(FLAGS.seed)

    saver = nn.ModuleDict({'policy': policy, 'normalizers': normalizers})
    print(saver)

    batch_size = FLAGS.BC.batch_size
    eval_gamma = 0.999
    for t in range(FLAGS.BC.max_iters):
        if t % FLAGS.BC.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(policy, env_eval)
            eval_returns_discount, eval_lengths_discount = evaluate(policy, env_eval, gamma=eval_gamma)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(
                    returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths))
                ), discounted_episode=dict(
                    returns=np.mean(eval_returns_discount), lengths=int(np.mean(eval_lengths_discount))
                )))

        expert_batch = expert_dataset.sample(batch_size)
        expert_state = np.stack([t.obs for t in expert_batch])
        expert_action = np.stack([t.action for t in expert_batch])
        _, loss, grad_norm = bc_loss.get_loss(expert_state, expert_action, fetch='train loss grad_norm')

        if FLAGS.BC.dagger and t % FLAGS.BC.collect_freq == 0 and t > 0:
            if t // FLAGS.BC.collect_freq == 1:
                collect_policy = expert_actor
                stochastic = False
                logger.info('Collect samples with expert actor...')
            else:
                collect_policy = policy
                stochastic = True
                logger.info('Collect samples with learned policy...')
            runner.reset()
            data, ep_infos = runner.run(collect_policy, FLAGS.BC.n_collect_samples, stochastic)
            data.action = expert_actor.get_actions(data.state, fetch='actions_mean')
            returns = [info['return'] for info in ep_infos]
            lengths = [info['length'] for info in ep_infos]
            for i in range(len(data)):
                expert_dataset.push_back(
                    data[i].state, data[i].action, data[i].next_state,
                    data[i].reward, data[i].mask, data[i].timeout
                )
            logger.info('Collect %d samples avg return = %.4f avg length = %d',
                        len(data), np.mean(returns), np.mean(lengths))
        if t % 100 == 0:
            mse_loss = policy.get_mse_loss(expert_state, expert_action)
            log_kvs(prefix='BC', kvs=dict(
                iter=t, loss=loss, grad_norm=grad_norm, mse_loss=mse_loss
            ))

    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())

    dict_result = dict()
    for gamma in [0.9, 0.99, 0.999, 1.0]:
        eval_returns, eval_lengths = evaluate(policy, env_eval, gamma=gamma)
        dict_result[gamma] = [float(np.mean(eval_returns)), eval_returns]
        logger.info('[%s]: %.4f', gamma, np.mean(eval_returns))

    save_path = os.path.join(FLAGS.log_dir, 'evaluate.yml')
    yaml.dump(dict_result, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()