import time
import os
import h5py
import shutil
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from utils import FLAGS, make_env, get_tf_config
from sac.policies.actor import Actor


def create_env(env_id, seed, log_dir, absorbing_state=False, rescale_action=True):
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


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    collect_mb = FLAGS.env.env_type == 'mb'
    if collect_mb:
        env_id = 'MB' + FLAGS.env.id
        logger.warning('Collect dataset for imitating environments')
    else:
        env_id = FLAGS.env.id
        logger.warning('Collect dataset for imitating policies')
    env = create_env(env_id, FLAGS.seed, FLAGS.log_dir, rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    actor = Actor(dim_state, dim_action, hidden_sizes=FLAGS.SAC.actor_hidden_sizes)

    tf.get_default_session().run(tf.global_variables_initializer())

    loader = nn.ModuleDict({'actor': actor})
    loader.load_state_dict(np.load(FLAGS.ckpt.policy_load, allow_pickle=True)[()])
    logger.info('Load policy from %s' % FLAGS.ckpt.policy_load)

    state_traj, action_traj, next_state_traj, reward_traj, len_traj = [], [], [], [], []
    returns = []
    while len(state_traj) < 50:
        states = np.zeros([env.max_episode_steps, dim_state], dtype=np.float32)
        actions = np.zeros([env.max_episode_steps, dim_action], dtype=np.float32)
        next_states = np.zeros([env.max_episode_steps, dim_state], dtype=np.float32)
        rewards = np.zeros([env.max_episode_steps], dtype=np.float32)
        state = env.reset()
        done = False
        t = 0
        while not done:
            action = actor.get_actions(state[None], fetch='actions_mean')
            next_state, reward, done, info = env.step(action)

            states[t] = state
            actions[t] = action
            rewards[t] = reward
            next_states[t] = next_state
            t += 1
            if done:
                break
            state = next_state
        if t < 700 or np.sum(rewards) < 0:
            continue
        state_traj.append(states)
        action_traj.append(actions)
        next_state_traj.append(next_states)
        reward_traj.append(rewards)
        len_traj.append(t)

        returns.append(np.sum(rewards))
        logger.info('# %d: collect a trajectory return = %.4f length = %d', len(state_traj), np.sum(rewards), t)

    state_traj = np.array(state_traj)
    action_traj = np.array(action_traj)
    next_state_traj = np.array(next_state_traj)
    reward_traj = np.array(reward_traj)
    len_traj = np.array(len_traj)
    assert len(state_traj.shape) == len(action_traj.shape) == 3
    assert len(reward_traj.shape) == 2 and len(len_traj.shape) == 1

    dataset = {
        'a_B_T_Da': action_traj,
        'len_B': len_traj,
        'obs_B_T_Do': state_traj,
        'r_B_T': reward_traj
    }
    if collect_mb:
        dataset['next_obs_B_T_Do'] = next_state_traj
    logger.info('Expert avg return = %.4f avg length = %d', np.mean(returns), np.mean(len_traj))

    if collect_mb:
        root_dir = 'dataset/mb2'
    else:
        root_dir = 'dataset/sac'

    save_dir = f'{root_dir}/{FLAGS.env.id}'
    os.makedirs(save_dir, exist_ok=True)
    shutil.copy(FLAGS.ckpt.policy_load, os.path.join(save_dir, 'policy.npy'))

    save_path = f'{root_dir}/{FLAGS.env.id}.h5'
    f = h5py.File(save_path, 'w')
    f.update(dataset)
    f.close()
    logger.info('save dataset into %s' % save_path)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
