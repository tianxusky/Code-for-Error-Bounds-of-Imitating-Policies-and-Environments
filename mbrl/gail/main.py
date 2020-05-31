import pickle
import os
import time
import yaml
import random
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.utils.normalizer import Normalizers
from sac.policies.actor import Actor
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from mbrl.gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from mbrl.gail.policies.gaussian_mlp_policy import GaussianMLPPolicy
from mbrl.gail.discriminator.discriminator import Discriminator
from mbrl.gail.v_function.mlp_v_function import MLPVFunction
from mbrl.gail.algos.trpo import TRPO
from mbrl.gail.utils.runner import VirtualRunner, evaluate as evaluate_on_virtual_env
from mbrl.bc.main import BehavioralCloningLoss
from utils.envs.mujoco.virtual_env import VirtualEnv
from gail.utils.runner import Runner, evaluate as evaluate_on_true_env
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


def set_random_seed(seed):
    assert seed > 0 and isinstance(seed, int)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = create_env(FLAGS.env.id, FLAGS.seed, rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

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

    # expert actor
    actor = Actor(dim_state, dim_action, hidden_sizes=FLAGS.SAC.actor_hidden_sizes)
    # generator
    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes,
                               output_diff=FLAGS.TRPO.output_diff, normalizers=normalizers)
    vfn = MLPVFunction(dim_state, dim_action, FLAGS.TRPO.vf_hidden_sizes, normalizers.state)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.algo.as_dict())

    subsampling_rate = env.max_episode_steps // FLAGS.GAIL.trajectory_size
    if FLAGS.GAIL.reward_type == 'nn':
        expert_batch = expert_dataset.buffer()
        expert_state = np.stack([t.obs for t in expert_batch])
        loc, scale = np.mean(expert_state, axis=0, keepdims=True), np.std(expert_state, axis=0, keepdims=True)
        del expert_batch, expert_state
        logger.info('loc = {}\nscale={}'.format(loc, scale))
        discriminator = Discriminator(dim_state, dim_action, normalizers=normalizers, subsampling_rate=subsampling_rate,
                                      loc=loc, scale=scale,
                                      **FLAGS.GAIL.discriminator.as_dict())
    else:
        raise NotImplementedError
    bc_loss = BehavioralCloningLoss(dim_state, dim_action, policy,
                                    lr=FLAGS.BC.lr, train_std=FLAGS.BC.train_std)
    tf.get_default_session().run(tf.global_variables_initializer())

    loader = nn.ModuleDict({'actor': actor})
    loader.load_state_dict(np.load(FLAGS.ckpt.policy_load, allow_pickle=True)[()])
    logger.info('Load policy from %s' % FLAGS.ckpt.policy_load)
    saver = nn.ModuleDict({'policy': policy, 'vfn': vfn, 'normalizers': normalizers, 'discriminator': discriminator})
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

    # pretrain
    for n_updates in range(FLAGS.GAIL.pretrain_iters):
        expert_batch = expert_dataset.sample(FLAGS.BC.batch_size)
        expert_state = np.stack([t.obs for t in expert_batch])
        expert_action = np.stack([t.action for t in expert_batch])
        expert_next_state = np.stack([t.next_obs for t in expert_batch])
        _, loss, grad_norm = bc_loss.get_loss(expert_state, expert_action, expert_next_state,
                                              fetch='train loss grad_norm')
        if n_updates % 100 == 0:
            mse_loss = policy.get_mse_loss(expert_state, expert_action, expert_next_state)
            logger.info('[Pretrain] iter = %d grad_norm = %.4f loss = %.4f mse_loss = %.4f',
                        n_updates, grad_norm, loss, mse_loss)

    # virtual env
    virtual_env = VirtualEnv(policy, env, n_envs=FLAGS.env.num_env, stochastic_model=True)
    virtual_runner = VirtualRunner(virtual_env, max_steps=env.max_episode_steps,
                                   gamma=FLAGS.TRPO.gamma, lambda_=FLAGS.TRPO.lambda_, rescale_action=False)
    env_eval_stochastic = VirtualEnv(policy, env, n_envs=4, stochastic_model=True)
    env_eval_deterministic = VirtualEnv(policy, env, n_envs=4, stochastic_model=False)

    max_ent_coef = FLAGS.TRPO.algo.ent_coef
    true_return = np.mean(eval_returns)
    for t in range(0, FLAGS.GAIL.total_timesteps, FLAGS.TRPO.rollout_samples*FLAGS.GAIL.g_iters):
        time_st = time.time()
        if t % FLAGS.GAIL.eval_freq == 0:
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
        # Generator
        generator_dataset = None
        for n_update in range(FLAGS.GAIL.g_iters):
            data, ep_infos = virtual_runner.run(actor, FLAGS.TRPO.rollout_samples, stochastic=False)
            # if FLAGS.TRPO.normalization:
            #     normalizers.state.update(data.state)
            #     normalizers.action.update(data.action)
            #     normalizers.diff.update(data.next_state - data.state)
            if t == 0:
                np.testing.assert_allclose(data.reward, env.mb_step(data.state, data.action, data.next_state)[0],
                                           atol=1e-4, rtol=1e-4)
            if t == 0 and n_update == 0 and not FLAGS.GAIL.learn_absorbing:
                data_ = data.copy()
                data_ = data_.reshape([FLAGS.TRPO.rollout_samples//env.n_envs, env.n_envs])
                for e in range(env.n_envs):
                    samples = data_[:, e]
                    masks = 1 - (samples.done | samples.timeout)[..., np.newaxis]
                    masks = masks[:-1]
                    assert np.allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)
            t += FLAGS.TRPO.rollout_samples
            data.reward = discriminator.get_reward(data.state, data.action, data.next_state)
            advantages, values = virtual_runner.compute_advantage(vfn, data)
            train_info = algo.train(max_ent_coef, data, advantages, values)
            fps = int(FLAGS.TRPO.rollout_samples / (time.time() - time_st))
            train_info['reward'] = np.mean(data.reward)
            train_info['fps'] = fps

            expert_batch = expert_dataset.sample(256)
            expert_state = np.stack([t.obs for t in expert_batch])
            expert_action = np.stack([t.action for t in expert_batch])
            expert_next_state = np.stack([t.next_obs for t in expert_batch])
            train_mse_loss = policy.get_mse_loss(expert_state, expert_action, expert_next_state)
            eval_mse_loss = policy.get_mse_loss(eval_state, eval_action, eval_next_state)
            train_info['mse_loss'] = dict(train=train_mse_loss, eval=eval_mse_loss)
            log_kvs(prefix='TRPO', kvs=dict(
                iter=t, **train_info
            ))

            generator_dataset = data

        # Discriminator
        for n_update in range(FLAGS.GAIL.d_iters):
            batch_size = FLAGS.GAIL.d_batch_size
            d_train_infos = dict()
            for generator_subset in generator_dataset.iterator(batch_size):
                expert_batch = expert_dataset.sample(batch_size)
                expert_state = np.stack([t.obs for t in expert_batch])
                expert_action = np.stack([t.action for t in expert_batch])
                expert_next_state = np.stack([t.next_obs for t in expert_batch])
                expert_mask = None
                train_info = discriminator.train(
                    expert_state, expert_action, expert_next_state,
                    generator_subset.state, generator_subset.action, generator_subset.next_state,
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

        if t % FLAGS.TRPO.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())
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
