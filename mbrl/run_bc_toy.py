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
from mbrl.gail.policies.gaussian_mlp_policy import GaussianMLPPolicy
from mbrl.gail.utils.runner import VirtualRunner, evaluate as evaluate_on_virtual_env
from gail.utils.runner import Runner, evaluate as evaluate_on_true_env
from utils.envs.mujoco.virtual_env import VirtualEnv
from utils import FLAGS, get_tf_config
from utils.envs.mujoco.linear_env import Actor


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


def collect_samples_from_true_env(env, actor, nb_episode=50, subsampling_rate=1, seed=2020):
    set_random_seed(seed)
    state_traj, action_traj, next_state_traj, reward_traj = [], [], [], []
    episode = 0
    while episode < nb_episode:
        state = env.reset()
        done = False
        t = 0
        return_ = 0
        while not done:
            action = actor.get_actions(state[None], fetch='actions_mean')[0]
            next_state, reward, done, info = env.step(action)
            return_ += reward
            state_traj.append(state)
            action_traj.append(action)
            next_state_traj.append(next_state)
            reward_traj.append(reward)
            t += 1
            if done:
                break
            state = next_state
        episode += 1
        logger.info('Collect a trajectory return = %.4f length = %d', return_, t)
    state_traj = np.array(state_traj)[::subsampling_rate]
    action_traj = np.array(action_traj)[::subsampling_rate]
    next_state_traj = np.array(next_state_traj)[::subsampling_rate]
    reward_traj = np.array(reward_traj)[::subsampling_rate]
    return state_traj, action_traj, next_state_traj, reward_traj


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

    # expert actor
    actor = Actor(dim_state, dim_action, init_std=0.)
    subsampling_rate = env.max_episode_steps // FLAGS.GAIL.trajectory_size
    expert_state, expert_action, expert_next_state, expert_reward = collect_samples_from_true_env(
        env=env, actor=actor, nb_episode=FLAGS.GAIL.traj_limit, subsampling_rate=subsampling_rate)
    logger.info('Collect % d samples avg return = %.4f', len(expert_state), np.mean(expert_reward))
    eval_state, eval_action, eval_next_state, eval_reward = collect_samples_from_true_env(
        env=env, actor=actor, nb_episode=3, seed=FLAGS.seed)
    loc, scale = np.mean(expert_state, axis=0, keepdims=True), np.std(expert_state, axis=0, keepdims=True)
    logger.info('loc = {}\nscale={}'.format(loc, scale))

    normalizers = Normalizers(dim_action=dim_action, dim_state=dim_state)
    policy = GaussianMLPPolicy(dim_state, dim_action, FLAGS.TRPO.policy_hidden_sizes,
                               output_diff=FLAGS.TRPO.output_diff, normalizers=normalizers)
    bc_loss = BehavioralCloningLoss(dim_state, dim_action, policy, lr=float(FLAGS.BC.lr), train_std=FLAGS.BC.train_std)

    tf.get_default_session().run(tf.global_variables_initializer())
    set_random_seed(FLAGS.seed)

    saver = nn.ModuleDict({'policy': policy, 'normalizers': normalizers})
    print(saver)

    # updater normalizer
    normalizers.state.update(expert_state)
    normalizers.action.update(expert_action)
    normalizers.diff.update(expert_next_state - expert_state)

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

        indices = np.random.randint(low=0, high=len(expert_state), size=batch_size)
        expert_state_ = expert_state[indices]
        expert_action_ = expert_action[indices]
        expert_next_state_ = expert_next_state[indices]
        _, loss, grad_norm = bc_loss.get_loss(expert_state_, expert_action_, expert_next_state_,
                                              fetch='train loss grad_norm')

        if t % 100 == 0:
            train_mse_loss = policy.get_mse_loss(expert_state_, expert_action_, expert_next_state_)
            eval_mse_loss = policy.get_mse_loss(eval_state, eval_action, eval_next_state)
            log_kvs(prefix='BC', kvs=dict(
                iter=t, grad_norm=grad_norm, loss=loss, mse_loss=dict(train=train_mse_loss, eval=eval_mse_loss)
            ))

    np.save('{}/final'.format(FLAGS.log_dir), saver.state_dict())

    dict_result = dict()
    for gamma in [0.9, 0.99, 0.999, 1.0]:
        eval_returns, eval_lengths = evaluate_on_virtual_env(actor, env_eval_deterministic, gamma=gamma)
        dict_result[gamma] = [float(np.mean(eval_returns)), eval_returns]
        logger.info('[%s]: %.4f', gamma, np.mean(eval_returns))

    save_path = os.path.join(FLAGS.log_dir, 'evaluate.yml')
    yaml.dump(dict_result, open(save_path, 'w'), default_flow_style=False)


if __name__ == '__main__':
    with tf.Session(config=get_tf_config()):
        main()
