
# This is an script to run gail using the dataset provided by Google.
set -e
set -x


ENV="LinearEnv-v2"
SEED=100
POLICY_HIDDEN_SIZES=100
BATCH_SIZE=128
LR=3e-4
MAX_ITERS=100000
TRAJ_LIMIT=30
TRAJ_SIZE=1000
OUTPUT_DIFF=False
TRAIN_STD=True


if [ "$(uname)" == "Darwin" ]; then
  python -m mbrl.run_bc_toy -s \
    algorithm="mbrl_bc_toy" \
    seed=${SEED} \
    env.id=${ENV} \
    env.env_type="mb" \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.output_diff=${OUTPUT_DIFF} \
    BC.batch_size=${BATCH_SIZE} \
    BC.lr=${LR} \
    BC.max_iters=${MAX_ITERS} \
    BC.train_std=${TRAIN_STD}
elif [ "$(uname)" == "Linux" ]; then
  for TRAJ_LIMIT in 3 10 30
  do
    for SEED in 100 200 300
    do
      python -m mbrl.run_bc_toy -s \
        algorithm="mbrl_bc_toy_${TRAJ_LIMIT}" \
        seed=${SEED} \
        env.id=${ENV} \
        env.env_type="mb" \
        GAIL.traj_limit=${TRAJ_LIMIT} \
        TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
        TRPO.output_diff=${OUTPUT_DIFF} \
        BC.batch_size=${BATCH_SIZE} \
        BC.lr=${LR} \
        BC.max_iters=${MAX_ITERS} \
        BC.train_std=${TRAIN_STD}  & sleep 2
    done
  done
  wait
fi
