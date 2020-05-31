import pickle
import os
import time
import yaml
import random
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.v_function.mlp_v_function import MLPVFunction
from trpo.algos.trpo import TRPO
from trpo.utils.normalizer import Normalizers
from gail.discriminator.discriminator import Discriminator
from gail.discriminator.linear_reward import LinearReward
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from gail.utils.runner import Runner, evaluate
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

    # load expert dataset
    subsampling_rate = env.max_episode_steps // FLAGS.GAIL.trajectory_size
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

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    vfn = MLPVFunction(dim_state, FLAGS.TRPO.vf_hidden_sizes, normalizers.state)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.algo.as_dict())

    if FLAGS.GAIL.reward_type == 'nn':
        expert_batch = expert_dataset.buffer()
        expert_state = np.stack([t.obs for t in expert_batch])
        loc, scale = np.mean(expert_state, axis=0, keepdims=True), np.std(expert_state, axis=0, keepdims=True)
        del expert_batch, expert_state
        discriminator = Discriminator(dim_state, dim_action, normalizers=normalizers, subsampling_rate=subsampling_rate,
                                      loc=loc, scale=scale,
                                      **FLAGS.GAIL.discriminator.as_dict())
    elif FLAGS.GAIL.reward_type in {'simplex', 'l2'}:
        discriminator = LinearReward(dim_state, dim_action, simplex=FLAGS.GAIL.reward_type == 'simplex')
    else:
        raise NotImplementedError
    tf.get_default_session().run(tf.global_variables_initializer())

    if not FLAGS.GAIL.reward_type == 'nn':
        expert_batch = expert_dataset.buffer()
        expert_state = np.stack([t.obs for t in expert_batch])
        expert_action = np.stack([t.action for t in expert_batch])
        discriminator.build(expert_state, expert_action)
        del expert_batch, expert_state, expert_action

    saver = nn.ModuleDict({'policy': policy, 'vfn': vfn, 'normalizers': normalizers, 'discriminator': discriminator})
    runner = Runner(env, max_steps=env.max_episode_steps, gamma=FLAGS.TRPO.gamma, lambda_=FLAGS.TRPO.lambda_,
                    add_absorbing_state=FLAGS.GAIL.learn_absorbing)
    print(saver)

    max_ent_coef = FLAGS.TRPO.algo.ent_coef
    eval_gamma = 0.999
    for t in range(0, FLAGS.GAIL.total_timesteps, FLAGS.TRPO.rollout_samples*FLAGS.GAIL.g_iters):
        time_st = time.time()
        if t % FLAGS.GAIL.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(policy, env_eval)
            eval_returns_discount, eval_lengths_discount = evaluate(policy, env_eval, gamma=eval_gamma)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(
                    returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths))
                ), discounted_episode=dict(
                    returns=np.mean(eval_returns_discount), lengths=int(np.mean(eval_lengths_discount))
                )))

        # Generator
        generator_dataset = None
        for n_update in range(FLAGS.GAIL.g_iters):
            data, ep_infos = runner.run(policy, FLAGS.TRPO.rollout_samples)
            if FLAGS.TRPO.normalization:
                normalizers.state.update(data.state)
                normalizers.action.update(data.action)
                normalizers.diff.update(data.next_state - data.state)
            if t == 0 and n_update == 0 and not FLAGS.GAIL.learn_absorbing:
                data_ = data.copy()
                data_ = data_.reshape([FLAGS.TRPO.rollout_samples//env.n_envs, env.n_envs])
                for e in range(env.n_envs):
                    samples = data_[:, e]
                    masks = 1 - (samples.done | samples.timeout)[..., np.newaxis]
                    masks = masks[:-1]
                    assert np.allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)
            t += FLAGS.TRPO.rollout_samples
            data.reward = discriminator.get_reward(data.state, data.action)
            advantages, values = runner.compute_advantage(vfn, data)
            train_info = algo.train(max_ent_coef, data, advantages, values)
            fps = int(FLAGS.TRPO.rollout_samples / (time.time() - time_st))
            train_info['reward'] = np.mean(data.reward)
            train_info['fps'] = fps

            expert_batch = expert_dataset.sample(256)
            expert_state = np.stack([t.obs for t in expert_batch])
            expert_action = np.stack([t.action for t in expert_batch])
            train_info['mse_loss'] = policy.get_mse_loss(expert_state, expert_action)
            log_kvs(prefix='TRPO', kvs=dict(
                iter=t, **train_info
            ))

            generator_dataset = data

        # Discriminator
        if FLAGS.GAIL.reward_type in {'nn', 'vb'}:
            for n_update in range(FLAGS.GAIL.d_iters):
                batch_size = FLAGS.GAIL.d_batch_size
                d_train_infos = dict()
                for generator_subset in generator_dataset.iterator(batch_size):
                    expert_batch = expert_dataset.sample(batch_size)
                    expert_state = np.stack([t.obs for t in expert_batch])
                    expert_action = np.stack([t.action for t in expert_batch])
                    expert_mask = np.stack([t.mask for t in expert_batch]).flatten() if FLAGS.GAIL.learn_absorbing else None
                    train_info = discriminator.train(
                        expert_state, expert_action,
                        generator_subset.state, generator_subset.action,
                        expert_mask,
                    )
                    for k, v in train_info.items():
                        if k not in d_train_infos:
                            d_train_infos[k] = []
                        d_train_infos[k].append(v)
                d_train_infos = {k: np.mean(v) for k, v in d_train_infos.items()}
                if n_update == FLAGS.GAIL.d_iters - 1:
                    log_kvs(prefix='Discriminator', kvs=dict(
                        iter=t, **d_train_infos
                    ))
        else:
            train_info = discriminator.train(generator_dataset.state, generator_dataset.action)
            log_kvs(prefix='Discriminator', kvs=dict(
                iter=t, **train_info
            ))

        if t % FLAGS.TRPO.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())
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