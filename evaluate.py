import pickle
import os
import time
import yaml
import tensorflow as tf
import numpy as np
import lunzi.nn as nn
from lunzi.Logger import logger, log_kvs
from trpo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from trpo.utils.normalizer import Normalizers
from gail.utils.runner import Runner, evaluate
# (TimeStep, ReplayBuffer) are required to restore from pickle.
from gail.utils.replay_buffer import TimeStep, ReplayBuffer, load_expert_dataset
from utils import FLAGS, get_tf_config


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

    tf.get_default_session().run(tf.global_variables_initializer())

    expert_result_path = os.path.join('logs', 'expert-%s.yml' % FLAGS.env.id)
    if not os.path.exists(expert_result_path):
        expert_dataset = load_expert_dataset(FLAGS.GAIL.buf_load)
        expert_reward = expert_dataset.get_average_reward()
        logger.info('Expert Reward %f', expert_reward)
        if FLAGS.GAIL.learn_absorbing:
            expert_dataset.add_absorbing_states(env)

        expert_result = dict()
        for gamma in [0.9, 0.99, 0.999, 1.0]:
            expert_returns = []
            discount = 1.
            expert_return = 0.
            for timestep in expert_dataset.buffer():
                expert_return += discount * timestep.reward[0]
                discount *= gamma
                if timestep.done:
                    expert_returns.append(float(expert_return))
                    discount = 1.
                    expert_return = 0.
            expert_result[gamma] = [float(np.mean(expert_returns)), expert_returns]
            logger.info('Expert gamma = %f %.4f (n_episode = %d)', gamma, np.mean(expert_returns), len(expert_returns))
        yaml.dump(expert_result,  open(expert_result_path, 'w'), default_flow_style=False)

    # loader policy
    loader = nn.ModuleDict({'policy': policy})
    root_dir = 'logs/gail_l2'
    for save_dir in sorted(os.listdir(root_dir)):
        if FLAGS.env.id not in save_dir:
            continue
        policy_load = os.path.join(root_dir, save_dir, 'stage-3000000.npy')
        loader.load_state_dict(np.load(policy_load, allow_pickle=True)[()])
        logger.warning('Load {} from {}'.format(loader.keys(), policy_load))

        dict_result = dict()
        for gamma in [0.9, 0.99, 0.999, 1.0]:
            eval_returns, eval_lengths = evaluate(policy, env_eval, gamma=gamma)
            dict_result[gamma] = [float(np.mean(eval_returns)), eval_returns]
            logger.info('[%s]: %.4f', gamma, np.mean(eval_returns))

        save_path = os.path.join(root_dir, save_dir, 'evaluate.yml')
        yaml.dump(dict_result,  open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()