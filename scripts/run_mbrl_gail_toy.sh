
# This is an script to run gail using the dataset provided by Google.
set -e
set -x

ENV="LinearEnv-v2"
NUM_ENV=1
SEED=100
ROLLOUT_SAMPLES=2000
VF_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
TRPO_ENT_COEF=0.0
G_ITERS=5
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=200
REWARD_TYPE="nn"
NEURAL_DISTANCE=False
GRADIENT_PENALTY_COEF=0.0
PRETRAIN_ITERS=0
TOTAL_TIMESTEPS=3000000


if [ "$(uname)" == "Darwin" ]; then
  python -m mbrl.run_gail_toy -s \
    algorithm="mbrl_gail_toy" \
    seed=${SEED} \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type="mb" \
    GAIL.g_iters=${G_ITERS} \
    GAIL.learn_absorbing=${LEARNING_ABSORBING} \
    GAIL.total_timesteps=${TOTAL_TIMESTEPS} \
    GAIL.pretrain_iters=${PRETRAIN_ITERS} \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    GAIL.reward_type=${REWARD_TYPE} \
    GAIL.discriminator.neural_distance=${NEURAL_DISTANCE} \
    GAIL.discriminator.gradient_penalty_coef=${GRADIENT_PENALTY_COEF} \
    TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
    TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.algo.ent_coef=${TRPO_ENT_COEF}
elif [ "$(uname)" == "Linux" ]; then
  for G_ITERS in 3 5
  do
    python -m mbrl.run_gail_toy -s \
      algorithm="mbrl_gail_toy" \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type="mb" \
      GAIL.g_iters=${G_ITERS} \
      GAIL.learn_absorbing=${LEARNING_ABSORBING} \
      GAIL.total_timesteps=${TOTAL_TIMESTEPS} \
      GAIL.pretrain_iters=${PRETRAIN_ITERS} \
      GAIL.traj_limit=${TRAJ_LIMIT} \
      GAIL.reward_type=${REWARD_TYPE} \
      GAIL.discriminator.neural_distance=${NEURAL_DISTANCE} \
      GAIL.discriminator.gradient_penalty_coef=${GRADIENT_PENALTY_COEF} \
      TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
      TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
      TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
      TRPO.algo.ent_coef=${TRPO_ENT_COEF}  & sleep 2
  done
  wait
fi
