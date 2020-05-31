
set -e
set -x

env="HalfCheetah-v2"
env_type="mb"
policy_load="logs/sac-HalfCheetah-v2-300-2020-05-20-22-07-48/final.npy"

if [ "$(uname)" == "Darwin" ]; then
  python collect.py -s \
    algorithm="collect" \
    env.id=$env \
    env.env_type=$env_type \
    ckpt.policy_load=$policy_load
elif [ "$(uname)" == "Linux" ]; then
  python collect.py -s \
    algorithm="collect" \
    env.id=$env \
    env.env_type=$env_type \
    ckpt.policy_load=$policy_load
fi
