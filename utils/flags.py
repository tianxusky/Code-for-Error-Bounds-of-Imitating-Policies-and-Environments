# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import time
import os
import sys
import yaml
from subprocess import check_output, CalledProcessError
from lunzi.config import BaseFLAGS, expand, parse
from lunzi.Logger import logger, FileSink, CSVWriter


class FLAGS(BaseFLAGS):
    _initialized = False

    seed = 100
    log_dir = None
    run_id = None
    algorithm = 'baseline'
    message = ''

    class env(BaseFLAGS):
        id = 'Hopper-v2'  # 'BreakoutNoFrameskip-v4'
        env_type = 'mujoco'    # 'atari'
        num_env = 1
        goal_env = False
        rescale_action = True  # only valid if env_type = mujoco

    class ACER(BaseFLAGS):
        gamma = 0.99
        q_coef = 0.5
        ent_coef = 0.01
        trust_region = True
        lr = 7e-4
        lrschedule = 'constant'

        n_steps = 20
        replay_ratio = 4
        replay_start = int(1e4)
        buffer_size = int(5e4)
        total_timesteps = int(1e6)
        save_freq = int(1e6)
        log_interval = 2000

    class TRPO(BaseFLAGS):
        gamma = 0.99
        lambda_ = 0.95

        vf_hidden_sizes = [64, 64]
        policy_hidden_sizes = [32, 32]

        total_timesteps = int(1e6)
        rollout_samples = 1000
        save_freq = int(0.5e6)
        eval_freq = int(1e4)
        normalization = True
        peb = False
        output_diff = False   # use for model-imitation

        class algo(BaseFLAGS):
            cg_damping = 0.1
            n_cg_iters = 10
            max_kl = 0.01
            vf_lr = 1e-3
            n_vf_iters = 5
            ent_coef = 0.00

        @classmethod
        def finalize(cls):
            if isinstance(cls.vf_hidden_sizes, int):
                cls.vf_hidden_sizes = [cls.vf_hidden_sizes] * 2
            if isinstance(cls.policy_hidden_sizes, int):
                cls.policy_hidden_sizes = [cls.policy_hidden_sizes] * 2

    class PPO(BaseFLAGS):
        gamma = 0.99
        lambda_ = 0.95
        reward_scale = 1.0

        vf_hidden_sizes = [64, 64]
        policy_hidden_sizes = [64, 64]

        total_timesteps = int(1e6)
        rollout_samples = 1000
        save_freq = int(1e6)
        eval_freq = int(1e4)
        normalization = True

        lr = 3e-4
        lr_schedule = 'linear'

        class algo(BaseFLAGS):
            clip_range = 0.2
            max_grad_norm = 0.5
            n_opt_epochs = 10
            ent_coef = 0.00

        @classmethod
        def finalize(cls):
            if isinstance(cls.vf_hidden_sizes, int):
                cls.vf_hidden_sizes = [cls.vf_hidden_sizes] * 2
            if isinstance(cls.policy_hidden_sizes, int):
                cls.policy_hidden_sizes = [cls.policy_hidden_sizes] * 2

    class SAC(BaseFLAGS):

        actor_hidden_sizes = [256, 256]
        critic_hidden_sizes = [256, 256]

        total_timesteps = int(1e6)
        init_random_steps = int(1e4)
        buffer_size = int(1e6)
        batch_size = 256
        target_entropy = None
        eval_freq = int(1e4)
        save_freq = int(1e6)
        log_freq = int(2e3)
        peb = True

        class algo(BaseFLAGS):
            gamma = 0.99

            actor_lr = 3e-4
            critic_lr = 3e-4
            alpha_lr = 3e-4

            target_update_freq = 1
            tau = 0.995
            actor_update_freq = 1

            init_alpha = 1.0
            learn_alpha = True

    class TD3(BaseFLAGS):
        actor_hidden_sizes = [256, 256]
        critic_hidden_sizes = [256, 256]

        total_timesteps = int(1e6)
        init_random_steps = int(10e3)
        buffer_size = int(1e6)
        batch_size = 256
        eval_freq = int(1e4)
        save_freq = int(1e6)
        log_freq = int(2e3)

        explore_noise = 0.1

        class algo(BaseFLAGS):
            gamma = 0.99

            actor_lr = 3e-4
            critic_lr = 3e-4

            policy_update_freq = 2
            policy_noise = 0.2
            policy_noise_clip = 0.5

            tau = 0.995

    class BC(BaseFLAGS):

        lr = 3e-4
        batch_size = 128
        eval_freq = 500
        train_std = True
        max_iters = int(1e4)

        dagger = False
        collect_freq = 5000
        n_collect_samples = 1000

    class GAIL(BaseFLAGS):
        total_timesteps = int(3e6)
        eval_freq = 1
        save_freq = 100
        g_iters = 5
        d_iters = 1
        reward_type = 'nn'
        learn_absorbing = False
        pretrain_iters = 0

        max_buf_size = int(1e6)
        d_batch_size = 64
        buf_load = None
        train_frac = 0.7
        traj_limit = 10
        trajectory_size = 50

        class discriminator(BaseFLAGS):
            hidden_sizes = [100, 100]
            lr = 3e-4
            ent_coef = 0.001
            max_grad_norm = None
            # Wasserstein distance parameters
            neural_distance = False
            gradient_penalty_coef = 0.
            l2_regularization_coef = 0.

        @classmethod
        def finalize(cls):
            if isinstance(cls.discriminator.hidden_sizes, int):
                cls.discriminator.hidden_sizes = [cls.discriminator.hidden_sizes] * 2

    class ckpt(BaseFLAGS):
        policy_load = None

    @classmethod
    def set_seed(cls):
        if cls.seed == 0:  # auto seed
            cls.seed = int.from_bytes(os.urandom(3), 'little') + 1  # never use seed 0 for RNG, 0 is for `urandom`
        logger.warning("Setting random seed to %s", cls.seed)

        import numpy as np
        import tensorflow as tf
        import random
        np.random.seed(cls.seed)
        tf.set_random_seed(cls.seed+1000)
        random.seed(cls.seed+2000)

    @classmethod
    def finalize(cls):
        log_dir = cls.log_dir
        if log_dir is None:
            run_id = cls.run_id
            if run_id is None:
                run_id = '{}-{}-{}-{}'.format(cls.algorithm, cls.env.id, cls.seed, time.strftime('%Y-%m-%d-%H-%M-%S'))

            log_dir = os.path.join("logs", run_id)
            cls.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        assert cls.TRPO.rollout_samples % cls.env.num_env == 0

        if os.path.exists('.git'):
            for t in range(10):
                try:
                    if sys.platform == 'linux':
                        cls.commit = check_output(['git', 'rev-parse', 'HEAD']).decode('utf-8').strip()
                        check_output(['git', 'add', '.'])
                        check_output(['git', 'checkout-index', '-a', '-f', '--prefix={}/src/'.format(cls.log_dir)])
                        open(os.path.join(log_dir, 'diff.patch'), 'w').write(
                            check_output(['git', '--no-pager', 'diff', 'HEAD']).decode('utf-8'))
                    else:
                        check_output(['git', 'checkout-index', '-a', '--prefix={}/src/'.format(cls.log_dir)])
                    break
                except Exception as e:
                    print(e)
                    print('Try again...')
                time.sleep(1)
            else:
                raise RuntimeError('Failed after 10 trials.')

        yaml.dump(cls.as_dict(), open(os.path.join(log_dir, 'config.yml'), 'w'), default_flow_style=False)
        # logger.add_sink(FileSink(os.path.join(log_dir, 'log.json')))
        logger.add_sink(FileSink(os.path.join(log_dir, 'log.txt')))
        logger.add_csvwriter(CSVWriter(os.path.join(log_dir, 'progress.csv')))
        logger.info("log_dir = %s", log_dir)

        cls.set_frozen()


parse(FLAGS)

