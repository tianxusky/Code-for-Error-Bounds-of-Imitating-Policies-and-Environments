
# This is an script to run gail using the dataset provided by Google.
set -e
set -x


ENV="HalfCheetah-v2"
SEED=100
POLICY_LOAD="dataset/mb2/${ENV}/policy.npy"
BUF_LOAD="dataset/mb2/${ENV}"
POLICY_HIDDEN_SIZES=100
BATCH_SIZE=128
LR=3e-4
MAX_ITERS=100000
TRAJ_LIMIT=3
TRAJ_SIZE=100
OUTPUT_DIFF=False
TRAIN_STD=True


if [ "$(uname)" == "Darwin" ]; then
  python -m mbrl.bc.main -s \
    algorithm="mbrl2_bc" \
    seed=${SEED} \
    env.id=${ENV} \
    env.env_type="mb" \
    GAIL.buf_load=${BUF_LOAD} \
    GAIL.traj_limit=${TRAJ_LIMIT} \
    GAIL.trajectory_size=${TRAJ_SIZE} \
    TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
    TRPO.output_diff=${OUTPUT_DIFF} \
    BC.batch_size=${BATCH_SIZE} \
    BC.lr=${LR} \
    BC.max_iters=${MAX_ITERS} \
    BC.train_std=${TRAIN_STD} \
    ckpt.policy_load=${POLICY_LOAD}
elif [ "$(uname)" == "Linux" ]; then
  for ENV in "Walker2d-v2" "HalfCheetah-v2" "Hopper-v2"
  do
    POLICY_LOAD="dataset/mb2/${ENV}/policy.npy"
    BUF_LOAD="dataset/mb2/${ENV}"
    for SEED in 100 200 300
    do
      python -m mbrl.bc.main -s \
        algorithm="mbrl2_bc" \
        seed=${SEED} \
        env.id=${ENV} \
        env.env_type="mb" \
        GAIL.buf_load=${BUF_LOAD} \
        GAIL.traj_limit=${TRAJ_LIMIT} \
        GAIL.trajectory_size=${TRAJ_SIZE} \
        TRPO.policy_hidden_sizes=${POLICY_HIDDEN_SIZES} \
        TRPO.output_diff=${OUTPUT_DIFF} \
        BC.batch_size=${BATCH_SIZE} \
        BC.lr=${LR} \
        BC.max_iters=${MAX_ITERS} \
        BC.train_std=${TRAIN_STD} \
        ckpt.policy_load=${POLICY_LOAD} & sleep 2
    done
  done
  wait
fi
