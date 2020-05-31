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


def main():
    FLAGS.set_seed()
    FLAGS.freeze()

    env = create_env(FLAGS.env.id, seed=FLAGS.seed, log_dir=FLAGS.log_dir, absorbing_state=FLAGS.GAIL.learn_absorbing,
                     rescale_action=FLAGS.env.rescale_action)
    dim_state = env.observation_space.shape[0]
    dim_action = env.action_space.shape[0]

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes, normalizer=normalizers.state)
    expert_actor = Actor(dim_state, dim_action, FLAGS.SAC.actor_hidden_sizes)
    tf.get_default_session().run(tf.global_variables_initializer())

    loader = nn.ModuleDict({'actor': expert_actor})
    actor_load = f'dataset/sac/{FLAGS.env.id}/policy.npy'
    loader.load_state_dict(np.load(actor_load, allow_pickle=True)[()])
    logger.warning('Load expert policy from %s' % actor_load)

    loader = nn.ModuleDict({'policy': policy})
    # policy_load = 'benchmarks/discounted-policies/bc/bc-Hopper-v2-100-2020-05-16-18-39-51/final.npy'
    policy_load = 'benchmarks/discounted-policies/gail_nn/gail-Hopper-v2-100-2020-05-17-00-50-42/final.npy'
    loader.load_state_dict(np.load(policy_load, allow_pickle=True)[()])
    logger.warning('Load policy from %s' % policy_load)

    for i in range(10):
        state = env.reset()
        return_ = 0.
        for t in range(env.max_episode_steps):
            env.render()
            action = expert_actor.get_actions(state[None], fetch='actions_mean')[0]

            next_state, reward, done, info = env.step(action)
            return_ += reward
            if done:
                break
            state = next_state
        print(return_)
    time.sleep(2)
    for i in range(10):
        state = env.reset()
        return_ = 0.
        for t in range(env.max_episode_steps):
            env.render()
            action = policy.get_actions(state[None], fetch='actions_mean')[0]

            next_state, reward, done, info = env.step(action)
            return_ += reward
            if done:
                break
            state = next_state
        print(return_)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()