
set -e
set -x

ENV=Hopper-v2
NUM_ENV=1


if [ "$(uname)" == "Darwin" ]; then
  python -m td3.main -s \
    algorithm=td3 \
    env.id=${ENV} \
    env.num_env=${NUM_ENV} \
    env.env_type=mujoco
elif [ "$(uname)" == "Linux" ]; then
  for ENV in Hopper-v2 HalfCheetah-v2 Ant-v2
  do
  for SEED in 100 200 300
    do
      python -m td3.main -s \
      algorithm=td3 \
      seed=${SEED} \
      env.id=${ENV} \
      env.num_env=${NUM_ENV} \
      env.env_type=mujoco  & sleep 2
    done
    wait
  done
fi
