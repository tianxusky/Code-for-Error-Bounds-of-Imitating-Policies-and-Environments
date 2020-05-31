# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
import collections
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from utils import FLAGS, make_env, get_tf_config
from trpo.utils.normalizer import Normalizers
from trpo.utils.runner import Runner, evaluate
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.v_function.mlp_v_function import MLPVFunction
from trpo.algos.trpo import TRPO


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=FLAGS.env.num_env, seed=FLAGS.seed, log_dir=FLAGS.log_dir,
                   rescale_action=FLAGS.env.rescale_action)
    env_eval = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=4, seed=FLAGS.seed+1000, log_dir=FLAGS.log_dir)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    vfn = MLPVFunction(dim_state, FLAGS.TRPO.vf_hidden_sizes, normalizers.state)
    algo = TRPO(vfn=vfn, policy=policy, dim_state=dim_state, dim_action=dim_action, **FLAGS.TRPO.algo.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())

    saver = nn.ModuleDict({'policy': policy, 'vfn': vfn, 'normalizers': normalizers})
    runner = Runner(env, max_steps=env.max_episode_steps, gamma=FLAGS.TRPO.gamma, lambda_=FLAGS.TRPO.lambda_,
                    partial_episode_bootstrapping=FLAGS.TRPO.peb)
    print(saver)

    max_ent_coef = FLAGS.TRPO.algo.ent_coef
    train_returns = collections.deque(maxlen=40)
    train_lengths = collections.deque(maxlen=40)
    for t in range(0, FLAGS.TRPO.total_timesteps, FLAGS.TRPO.rollout_samples):
        time_st = time.time()
        if t % FLAGS.TRPO.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(policy, env_eval)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths)))
            ))

        data, ep_infos = runner.run(policy, FLAGS.TRPO.rollout_samples)
        if t == 0:
            data_ = data.copy()
            data_ = data_.reshape([FLAGS.TRPO.rollout_samples//env.n_envs, env.n_envs])
            for e in range(env.n_envs):
                samples = data_[:, e]
                masks = 1 - (samples.done | samples.timeout)[..., np.newaxis]
                masks = masks[:-1]
                assert np.allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)

        if FLAGS.TRPO.normalization:
            normalizers.state.update(data.state)
            normalizers.action.update(data.action)
            normalizers.diff.update(data.next_state - data.state)
        advantages, values = runner.compute_advantage(vfn, data)
        train_info = algo.train(max_ent_coef, data, advantages, values)
        train_returns.extend([info['return'] for info in ep_infos])
        train_lengths.extend([info['length'] for info in ep_infos])
        fps = int(FLAGS.TRPO.rollout_samples / (time.time() - time_st))
        train_info['fps'] = fps
        log_kvs(prefix='TRPO', kvs=dict(
            iter=t, episode=dict(
                returns=np.mean(train_returns) if len(train_returns) > 0 else 0.,
                lengths=int(np.mean(train_lengths) if len(train_lengths) > 0 else 0)),
            **train_info))

        t += FLAGS.TRPO.rollout_samples
        if t % FLAGS.TRPO.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())
    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
