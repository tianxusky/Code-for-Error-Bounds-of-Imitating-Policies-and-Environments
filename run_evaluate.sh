set -e
set -x

ENV=Hopper-v2
NUM_ENV=1
ROLLOUT_SAMPLES=1000
BUF_LOAD=dataset/sac/${ENV}
POLICY_HIDDEN_SIZES=100


python evaluate.py -s \
  algorithm="test" \
  env.id=${ENV} \
  env.num_env=${NUM_ENV} \
  env.env_type=mujoco \
  GAIL.buf_load=${BUF_LOAD} \
  TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
  TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES}

