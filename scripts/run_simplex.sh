
# This is an script to run gail using the dataset provided by Google.
set -e
set -x

ENV=Walker2d-v2
NUM_ENV=1
SEED=200
ROLLOUT_SAMPLES=1000
BUF_LOAD=dataset/sac/${ENV}
VF_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
TRPO_ENT_COEF=0.0
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=1000
REWARD_TYPE="simplex"

if [ "$(uname)" == "Darwin" ]; then
  python -m gail.main -s \
    algorithm=${REWARD_TYPE} \
    seed=${SEED} \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco \
    GAIL.buf_load=${BUF_LOAD} \
    GAIL.learn_absorbing=${LEARNING_ABSORBING} \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    GAIL.reward_type=${REWARD_TYPE} \
    TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
    TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.algo.ent_coef=${TRPO_ENT_COEF}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "Hopper-v2" "Walker2d-v2" "HalfCheetah-v2"
  do
    BUF_LOAD=dataset/sac/${ENV}
    for SEED in 100 200 300
    do
     python -m gail.main -s \
        algorithm=${REWARD_TYPE} \
        seed=${SEED} \
        env.id=${ENV} \
        env.num_env=${NUM_ENV} \
        env.env_type=mujoco \
        GAIL.buf_load=${BUF_LOAD} \
        GAIL.learn_absorbing=${LEARNING_ABSORBING} \
        GAIL.traj_limit=${TRAJ_LIMIT} \
        GAIL.trajectory_size=${TRAJ_SIZE} \
        GAIL.reward_type=${REWARD_TYPE} \
        TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
        TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
        TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
        TRPO.algo.ent_coef=${TRPO_ENT_COEF} & sleep 2
     done
     wait
  done
fi
