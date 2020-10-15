

set -e
set -x

ENV=Hopper-v2
NUM_ENV=1
SEED=200
BUF_LOAD=dataset/sac/${ENV}
POLICY_LOAD="dataset/sac/${ENV}/policy.npy"
VF_HIDDEN_SIZES=100
D_HIDDEN_SIZES=100
POLICY_HIDDEN_SIZES=100
# D
POLICY_ENT_COEF=0.1
# Learning
TRPO_ENT_COEF=0.0
LEARNING_ABSORBING=False
TRAJ_LIMIT=3
TRAJ_SIZE=1000
ROLLOUT_SAMPLES=1000
TOTAL_TIMESTEPS=3000000
if [ "$(uname)" == "Darwin" ]; then
    python -m airl.main -s \
      algorithm="airl" \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=mujoco \
      ckpt.policy_load=${POLICY_LOAD} \
      AIRL.buf_load=${BUF_LOAD} \
      AIRL.learn_absorbing=${LEARNING_ABSORBING} \
      AIRL.traj_limit=${TRAJ_LIMIT} \
      AIRL.trajectory_size=${TRAJ_SIZE} \
      AIRL.discriminator.hidden_sizes=${D_HIDDEN_SIZES} \
      AIRL.discriminator.policy_ent_coef=${POLICY_ENT_COEF} \
      AIRL.total_timesteps=${TOTAL_TIMESTEPS} \
      TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
      TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
      TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
      TRPO.algo.ent_coef=${TRPO_ENT_COEF}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "Hopper-v2" "Walker2d-v2" "HalfCheetah-v2"
  do
    BUF_LOAD=dataset/sac/${ENV}
    POLICY_LOAD="dataset/sac/${ENV}/policy.npy"
    for SEED in 100 200 300
    do
    python -m airl.main -s \
      algorithm="airl" \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=mujoco \
      ckpt.policy_load=${POLICY_LOAD} \
      AIRL.buf_load=${BUF_LOAD} \
      AIRL.learn_absorbing=${LEARNING_ABSORBING} \
      AIRL.traj_limit=${TRAJ_LIMIT} \
      AIRL.trajectory_size=${TRAJ_SIZE} \
      AIRL.discriminator.hidden_sizes=${D_HIDDEN_SIZES} \
      AIRL.discriminator.policy_ent_coef=${POLICY_ENT_COEF} \
      AIRL.total_timesteps=${TOTAL_TIMESTEPS} \
      TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
      TRPO.vf_hidden_sizes=${VF_HIDDEN_SIZES} \
      TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
      TRPO.algo.ent_coef=${TRPO_ENT_COEF} & sleep 2
     done
    wait
  done
fi

