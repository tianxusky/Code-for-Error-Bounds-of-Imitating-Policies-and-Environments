
# This is an script to run gail using the dataset provided by Google.
set -e
set -x

ENV=Walker2d-v2
NUM_ENV=1
SEED=200
BUF_LOAD=dataset/sac/${ENV}
VF_HIDDEN_SIZES=100
D_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
# Discriminator
NEURAL_DISTANCE=True
GRADIENT_PENALTY_COEF=10.0
L2_REGULARIZATION_COEF=0.0
REWARD_TYPE="nn"
# Learning
TRPO_ENT_COEF=0.0
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=1000
ROLLOUT_SAMPLES=1000
TOTAL_TIMESTEPS=3000000

if [ "$(uname)" == "Darwin" ]; then
  python -m gail.main -s \
    algorithm="gail" \
    seed=${SEED} \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco \
    GAIL.buf_load=${BUF_LOAD} \
    GAIL.learn_absorbing=${LEARNING_ABSORBING} \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    GAIL.reward_type=${REWARD_TYPE} \
    GAIL.discriminator.neural_distance=${NEURAL_DISTANCE} \
    GAIL.discriminator.hidden_sizes=${D_HIDDEN_SIZES} \
    GAIL.discriminator.gradient_penalty_coef=${GRADIENT_PENALTY_COEF} \
    GAIL.discriminator.l2_regularization_coef=${L2_REGULARIZATION_COEF} \
    GAIL.total_timesteps=${TOTAL_TIMESTEPS} \
    TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
    TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.algo.ent_coef=${TRPO_ENT_COEF}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "Walker2d-v2" "HalfCheetah-v2" "Hopper-v2"
  do
    BUF_LOAD=dataset/sac/${ENV}
    for SEED in 100 200 300
    do
      python -m gail.main -s \
        algorithm="gail_w" \
        seed=${SEED} \
        env.id=${ENV} \
        env.num_env=${NUM_ENV} \
        env.env_type=mujoco \
        GAIL.buf_load=${BUF_LOAD} \
        GAIL.learn_absorbing=${LEARNING_ABSORBING} \
        GAIL.traj_limit=${TRAJ_LIMIT} \
        GAIL.trajectory_size=${TRAJ_SIZE} \
        GAIL.reward_type=${REWARD_TYPE} \
        GAIL.discriminator.neural_distance=${NEURAL_DISTANCE} \
        GAIL.discriminator.hidden_sizes=${D_HIDDEN_SIZES} \
        GAIL.discriminator.gradient_penalty_coef=${GRADIENT_PENALTY_COEF} \
        GAIL.discriminator.l2_regularization_coef=${L2_REGULARIZATION_COEF} \
        GAIL.total_timesteps=${TOTAL_TIMESTEPS} \
        TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
        TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
        TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
        TRPO.algo.ent_coef=${TRPO_ENT_COEF} & sleep 2
     done
  done
  wait
fi
