# Error Bounds of Imitating Policies and Environments

This is the repository hosting the code used for the paper Error Bounds of Imitating Policies and Environments. The code contains the implementation of the BC, GAIL, DAgger, FEM, MWAL, MBRL_BC, MBRL_GAIL.

## Requirements

We use Python 3.6 to run all experiments. Please install MuJoCo following the instructions from [mujoco-py](https://github.com/openai/mujoco-py). Other python packages are listed in [requirement.txt](requirement.txt)

## Dataset

Dataset, including expert demonstrations and expert policies (parameters), is provided in the folder of [dataset](dataset).

However, one can run SAC to re-train expert policies (see [scripts/run_sac.sh](scripts/run_sac.sh)) and to collect expert demonstrations (see [scripts/run_collect.sh](scripts/run_collect.sh)).

## Usage

The folder of [scripts](scripts) provides all demo running scripts to test algorithms like GAIL, BC, DAgger, FEM, GTAL, and imitating-environments algorithms.