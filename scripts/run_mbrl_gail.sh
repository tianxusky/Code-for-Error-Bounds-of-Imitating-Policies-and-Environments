
# This is an script to run gail using the dataset provided by Google.
set -e
set -x

ENV="Walker2d-v2"
NUM_ENV=1
SEED=100
ROLLOUT_SAMPLES=1000
POLICY_LOAD="dataset/mb2/${ENV}/policy.npy"
BUF_LOAD="dataset/mb2/${ENV}"
VF_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
TRPO_ENT_COEF=0.0
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=1000
REWARD_TYPE="nn"
NEURAL_DISTANCE=False
GRADIENT_PENALTY_COEF=1.0
PRETRAIN_ITERS=0
TOTAL_TIMESTEPS=3000000


if [ "$(uname)" == "Darwin" ]; then
  python -m mbrl.gail.main -s \
    algorithm="mbrl_gail" \
    seed=${SEED} \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type="mb" \
    GAIL.buf_load=${BUF_LOAD} \
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
    TRPO.algo.ent_coef=${TRPO_ENT_COEF} \
    ckpt.policy_load=${POLICY_LOAD}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "HalfCheetah-v2" "Walker2d-v2" "Hopper-v2"
  do
    POLICY_LOAD="dataset/mb2/${ENV}/policy.npy"
    BUF_LOAD="dataset/mb2/${ENV}"
    for SEED in 100 200 300
    do
      python -m mbrl.gail.main -s \
        algorithm="mbrl2_gail" \
        seed=${SEED} \
        env.id=${ENV} \
        env.num_env=${NUM_ENV} \
        env.env_type="mb" \
        GAIL.buf_load=${BUF_LOAD} \
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
        TRPO.algo.ent_coef=${TRPO_ENT_COEF} \
        ckpt.policy_load=${POLICY_LOAD} & sleep 2
     done
  done
  wait
fi
