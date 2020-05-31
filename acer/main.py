import time
import collections
import tensorflow as tf
import numpy as np
from lunzi import nn
from lunzi.Logger import logger, log_kvs
from acer.policies.cnn_policy import CNNPolicy
from acer.policies.mlp_policy import MLPPolicy
from acer.algos.acer import ACER
from acer.utils.runner import Runner, gen_dtype
from acer.utils.buffer import ReplayBuffer
from utils import FLAGS, get_tf_config, make_env


def check_data_equal(src, dst, attributes):
    for attr in attributes:
        np.testing.assert_allclose(getattr(src, attr), getattr(dst, attr), err_msg='%s is not equal' % attr)


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=FLAGS.env.num_env, seed=FLAGS.seed, log_dir=FLAGS.log_dir)
    state_spec = env.observation_space
    action_spec = env.action_space

    logger.info('[{}]: state_spec:{}, action_spec:{}'.format(FLAGS.env.id, state_spec.shape, action_spec.n))

    dtype = gen_dtype(env, 'state action next_state mu reward done timeout info')
    buffer = ReplayBuffer(env.n_envs, FLAGS.ACER.n_steps, stacked_frame=FLAGS.env.env_type == 'atari',
                          dtype=dtype, size=FLAGS.ACER.buffer_size)

    if len(state_spec.shape) == 3:
        policy = CNNPolicy(state_spec, action_spec)
    else:
        policy = MLPPolicy(state_spec, action_spec)

    algo = ACER(state_spec, action_spec, policy, lr=FLAGS.ACER.lr, lrschedule=FLAGS.ACER.lrschedule,
                total_timesteps=FLAGS.ACER.total_timesteps, ent_coef=FLAGS.ACER.ent_coef, q_coef=FLAGS.ACER.q_coef,
                trust_region=FLAGS.ACER.trust_region)
    runner = Runner(env, max_steps=env.max_episode_steps, gamma=FLAGS.ACER.gamma)
    saver = nn.ModuleDict({'policy': policy})
    print(saver)

    tf.get_default_session().run(tf.global_variables_initializer())
    algo.update_old_policy(0.)

    n_steps = FLAGS.ACER.n_steps
    n_batches = n_steps * env.n_envs
    n_stages = FLAGS.ACER.total_timesteps // n_batches

    returns = collections.deque(maxlen=40)
    lengths = collections.deque(maxlen=40)
    replay_reward = collections.deque(maxlen=40)
    time_st = time.time()
    for t in range(n_stages):
        data, ep_infos = runner.run(policy, n_steps)
        returns.extend([info['return'] for info in ep_infos])
        lengths.extend([info['length'] for info in ep_infos])

        if t == 0:  # check runner
            indices = np.arange(0, n_batches, env.n_envs)
            for _ in range(env.n_envs):
                samples = data[indices]
                masks = 1 - (samples.done | samples.timeout)
                masks = masks[:-1]
                masks = np.reshape(masks, [-1] + [1] * len(samples.state.shape[1:]))
                np.testing.assert_allclose(samples.state[1:] * masks, samples.next_state[:-1] * masks)
                indices += 1

        buffer.store_episode(data)
        if t == 1:  # check buffer
            data_ = buffer.sample(idx=[1 for _ in range(env.n_envs)])
            check_data_equal(data_, data, ('state', 'action', 'next_state', 'mu', 'reward', 'done', 'timeout'))

        # on-policy training
        qret = runner.compute_qret(policy, data)
        train_info = algo.train(data, qret, t*n_batches)
        replay_reward.append(np.mean(data.reward))
        # off-policy training
        if t*n_batches > FLAGS.ACER.replay_start:
            n = np.random.poisson(FLAGS.ACER.replay_ratio)
            for _ in range(n):
                data = buffer.sample()
                qret = runner.compute_qret(policy, data)
                algo.train(data, qret, t*n_batches)
                replay_reward.append(np.mean(data.reward))

        if t*n_batches % FLAGS.ACER.log_interval == 0:
            fps = int(t*n_batches / (time.time()-time_st))
            kvs = dict(iter=t*n_batches, episode=dict(
                            returns=np.mean(returns) if len(returns) > 0 else 0,
                            lengths=np.mean(lengths).astype(np.int32) if len(lengths) > 0 else 0),
                       **train_info,
                       replay_reward=np.mean(replay_reward) if len(replay_reward) > 0 else 0.,
                       fps=fps)
            log_kvs(prefix='ACER', kvs=kvs)

        if t*n_batches % FLAGS.ACER.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())
    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()) as sess:
        main()
