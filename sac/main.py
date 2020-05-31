import time
import collections
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.dataset import Dataset
from lunzi.Logger import logger, log_kvs
from utils import FLAGS, make_env, get_tf_config
from trpo.utils.runner import gen_dtype, evaluate
from sac.policies.critic import Critic
from sac.policies.actor import Actor
from sac.algos.sac import SAC


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=FLAGS.env.num_env, seed=FLAGS.seed, log_dir=FLAGS.log_dir,
                   rescale_action=FLAGS.env.rescale_action)
    env_eval = make_env(FLAGS.env.id, FLAGS.env.env_type, num_env=4, seed=FLAGS.seed+1000, log_dir=FLAGS.log_dir)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    actor = Actor(dim_state, dim_action, hidden_sizes=FLAGS.SAC.actor_hidden_sizes)
    critic = Critic(dim_state, dim_action, hidden_sizes=FLAGS.SAC.critic_hidden_sizes)
    target_entropy = FLAGS.SAC.target_entropy
    if target_entropy is None:
        target_entropy = - dim_action
    sac = SAC(dim_state, dim_action, actor=actor, critic=critic, target_entropy=target_entropy, **FLAGS.SAC.algo.as_dict())

    tf.get_default_session().run(tf.global_variables_initializer())
    sac.update_critic_target(tau=0.0)

    dtype = gen_dtype(env, 'state action next_state reward done')
    buffer = Dataset(dtype=dtype, max_size=FLAGS.SAC.buffer_size)
    saver = nn.ModuleDict({'actor': actor, 'critic': critic})
    print(saver)

    n_steps = np.zeros(env.n_envs)
    n_returns = np.zeros(env.n_envs)

    train_returns = collections.deque(maxlen=40)
    train_lengths = collections.deque(maxlen=40)
    states = env.reset()
    time_st = time.time()
    for t in range(FLAGS.SAC.total_timesteps):
        if t < FLAGS.SAC.init_random_steps:
            actions = np.array([env.action_space.sample() for _ in range(env.n_envs)])
        else:
            actions = actor.get_actions(states)
        next_states, rewards, dones, infos = env.step(actions)
        n_returns += rewards
        n_steps += 1
        timeouts = n_steps == env.max_episode_steps
        terminals = np.copy(dones)
        for e, info in enumerate(infos):
            if FLAGS.SAC.peb and info.get('TimeLimit.truncated', False):
                terminals[e] = False

        transitions = [states, actions, next_states.copy(), rewards, terminals]
        buffer.extend(np.rec.fromarrays(transitions, dtype=dtype))

        indices = np.where(dones | timeouts)[0]
        if len(indices) > 0:
            next_states[indices] = env.partial_reset(indices)

            train_returns.extend(n_returns[indices])
            train_lengths.extend(n_steps[indices])
            n_returns[indices] = 0
            n_steps[indices] = 0
        states = next_states.copy()

        if t >= FLAGS.SAC.init_random_steps:
            samples = buffer.sample(FLAGS.SAC.batch_size)
            train_info = sac.train(samples)
            if t % FLAGS.SAC.log_freq == 0:
                fps = int(t / (time.time() - time_st))
                train_info['fps'] = fps
                log_kvs(prefix='SAC', kvs=dict(
                    iter=t, episode=dict(
                        returns=np.mean(train_returns) if len(train_returns) > 0 else 0.,
                        lengths=int(np.mean(train_lengths) if len(train_lengths) > 0 else 0)),
                    **train_info))

        if t % FLAGS.SAC.eval_freq == 0:
            eval_returns, eval_lengths = evaluate(actor, env_eval)
            log_kvs(prefix='Evaluate', kvs=dict(
                iter=t, episode=dict(returns=np.mean(eval_returns), lengths=int(np.mean(eval_lengths)))
            ))

        if t % FLAGS.SAC.save_freq == 0:
            np.save('{}/stage-{}'.format(FLAGS.log_dir, t), saver.state_dict())
            np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())

    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
