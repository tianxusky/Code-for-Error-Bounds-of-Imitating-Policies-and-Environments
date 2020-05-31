__all__ = ['generate_data', 'generate_new_param_values']

import tensorflow as tf
import numpy as np


def generate_data(observation_space, action_space, n_env_, n_step_, seed=None, verbose=False):
    try:
        action_space.seed(seed)
    except AttributeError:
        pass
    np.random.seed(seed)
    print('seed:{}, uniform:{}'.format(seed, np.random.uniform()))
    state_, action_, reward_, done_, mu_ = [], [], [], [], []
    current_state = np.random.randn(*[n_env_, *observation_space.shape]) * 0.01
    for _ in range(n_step_):
        state_.append(current_state)
        action_.append(np.random.randint(low=0, high=action_space.n, size=[n_env_]))
        reward_.append(np.random.randn(*[n_env_]))
        _mu = np.random.uniform(size=[n_env_, action_space.n])
        mu_.append(_mu / np.sum(_mu, axis=-1, keepdims=True))
        terminal = [False for _ in range(n_env_)]
        for i in range(n_env_):
            if np.random.uniform() < 0.1:
                terminal[i] = True
        done_.append(terminal)
        current_state = np.random.randn(*[n_env_, *observation_space.shape]) * 0.01
    state_.append(current_state)

    state_ = np.array(state_)
    action_ = np.array(action_)
    reward_ = np.array(reward_)
    done_ = np.array(done_)
    mu_ = np.array(mu_)

    if verbose:
        print('state mean:{}, std:{}'.format(np.mean(state_), np.std(state_)))
        print('action mean:{}, std:{}'.format(np.mean(action_), np.std(action_)))
        print('reward mean:{}, std:{}'.format(np.mean(reward_), np.std(reward_)))
        print('done mean:{}, std:{}'.format(np.mean(done_), np.std(done_)))
        print('mu mean:{}, std:{}'.format(np.mean(mu_), np.std(mu_)))

    assert state_.shape[:2] == (n_step_ + 1, n_env_)
    assert action_.shape[:2] == reward_.shape[:2] == done_.shape[:2] == mu_.shape[:2] == (n_step_, n_env_)
    return state_, action_, reward_, done_, mu_


def generate_new_param_values(params_, seed=None):
    np.random.seed(seed)
    new_values_ = []
    for param in params_:
        new_values_.append(np.random.randn(*param.get_shape().as_list()) * 0.01)
    return new_values_


def check_shape(ts,shapes):
    i = 0
    for (t,shape) in zip(ts,shapes):
        assert t.get_shape().as_list()==shape, "id " + str(i) + " shape " + str(t.get_shape()) + str(shape)
        i += 1


def seq_to_batch(h, flat=False):
    shape = h[0].get_shape().as_list()
    if not flat:
        assert (len(shape) > 1)
        nh = h[0].get_shape()[-1].value
        return tf.reshape(tf.concat(axis=1, values=h), [-1, nh])
    else:
        return tf.reshape(tf.stack(values=h, axis=1), [-1])


# remove last step
def strip(var, nenvs, nsteps, flat=False):
    vars = batch_to_seq(var, nenvs, nsteps + 1, flat)
    return seq_to_batch(vars[:-1], flat)


def q_retrace(R, D, q_i, v, rho_i, nenvs, nsteps, gamma):
    """
    Calculates q_retrace targets

    :param R: Rewards
    :param D: Dones
    :param q_i: Q values for actions taken
    :param v: V values
    :param rho_i: Importance weight for each action
    :return: Q_retrace values
    """
    rho_bar = batch_to_seq(tf.minimum(1.0, rho_i), nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    rs = batch_to_seq(R, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    ds = batch_to_seq(D, nenvs, nsteps, True)  # list of len steps, shape [nenvs]
    q_is = batch_to_seq(q_i, nenvs, nsteps, True)
    vs = batch_to_seq(v, nenvs, nsteps + 1, True)
    v_final = vs[-1]
    qret = v_final
    qrets = []
    for i in range(nsteps - 1, -1, -1):
        check_shape([qret, ds[i], rs[i], rho_bar[i], q_is[i], vs[i]], [[nenvs]] * 6)
        qret = rs[i] + gamma * qret * (1.0 - ds[i])
        qrets.append(qret)
        qret = (rho_bar[i] * (qret - q_is[i])) + vs[i]
    qrets = qrets[::-1]
    qret = seq_to_batch(qrets, flat=True)
    return qret


def batch_to_seq(h, nbatch, nsteps, flat=False):
    if flat:
        h = tf.reshape(h, [nbatch, nsteps])
    else:
        h = tf.reshape(h, [nbatch, nsteps, -1])
    return [tf.squeeze(v, [1]) for v in tf.split(axis=1, num_or_size_splits=nsteps, value=h)]


def test(_):
    tf.set_random_seed(100)
    np.random.seed(100)
    sess = tf.Session()
    n_env, n_step = 2, 20
    gamma = 0.99

    R = tf.placeholder(tf.float32, [n_env*n_step])
    D = tf.placeholder(tf.float32, [n_env*n_step])
    q_i = tf.placeholder(tf.float32, [n_env*n_step])
    v = tf.placeholder(tf.float32, [n_env*(n_step+1)])
    rho_i = tf.placeholder(tf.float32, [n_env*n_step])

    qret = q_retrace(R, D, q_i, v, rho_i, n_env, n_step, gamma)

    td_map = {
        R: np.random.randn(*[n_env*n_step]),
        D: np.zeros(*[n_env*n_step]),
        q_i: np.random.randn(*[n_env*n_step]),
        v: np.random.randn(*[n_env*(n_step+1)]),
        rho_i: np.random.randn(*[n_env*n_step])
    }
    res = sess.run(qret, feed_dict=td_map)
    print(res)

if __name__ == '__main__':
    test('')
