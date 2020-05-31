
# This is an script to run gail using the dataset provided by Google.
set -e
set -x


ENV=Walker2d-v2
SEED=100
BUF_LOAD=dataset/sac/${ENV}
POLICY_HIDDEN_SIZES=100
BATCH_SIZE=128
LR=3e-4
MAX_ITERS=100000
TRAJ_LIMIT=3
TRAJ_SIZE=1000
TRAIN_STD=True
DAGGER=False


if [ "$(uname)" == "Darwin" ]; then
  python -m gail.bc -s \
    algorithm="bc" \
    seed=${SEED} \
    env.id=${ENV} \
    env.env_type=mujoco \
    GAIL.buf_load=${BUF_LOAD} \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    BC.batch_size=${BATCH_SIZE} \
    BC.lr=${LR} \
    BC.max_iters=${MAX_ITERS} \
    BC.train_std=${TRAIN_STD} \
    BC.dagger=${DAGGER}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "Hopper-v2" "Walker2d-v2" "HalfCheetah-v2"
  do
    BUF_LOAD=dataset/sac/${ENV}
    for SEED in 100 200 300
    do
    python -m gail.bc -s \
      algorithm="bc" \
      seed=${SEED} \
      env.id=${ENV} \
      env.env_type=mujoco \
      GAIL.buf_load=${BUF_LOAD} \
      GAIL.traj_limit=${TRAJ_LIMIT} \
      GAIL.trajectory_size=${TRAJ_SIZE} \
      TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
      BC.batch_size=${BATCH_SIZE} \
      BC.lr=${LR} \
      BC.max_iters=${MAX_ITERS} \
      BC.train_std=${TRAIN_STD} & sleep 2
    done
    wait
  done
fi
