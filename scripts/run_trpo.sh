
set -e
set -x

ENV=HalfCheetah-v2
NUM_ENV=1
ROLLOUT_SAMPLES=1000
PEB=False
TOTAL_TIMESTEPS=3000000


if [ "$(uname)" == "Darwin" ]; then
  python -m trpo.main -s \
    algorithm=trpo \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco \
    TRPO.rollout_samples=${ROLLOUT_SAMPLES} \
    TRPO.peb=${PEB} \
    TRPO.total_timesteps=${TOTAL_TIMESTEPS}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in Hopper-v2 HalfCheetah-v2 Ant-v2
  do
    for SEED in 100 200 300
    do
      python -m trpo.main -s \
      algorithm=trpo \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=mujoco \
      TRPO.rollout_samples=${ROLLOUT_SAMPLES}  \
      TRPO.peb=${PEB} \
      TRPO.total_timesteps=${TOTAL_TIMESTEPS} & sleep 2
    done
    wait
  done
fi
