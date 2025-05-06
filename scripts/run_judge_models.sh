#!/bin/bash -x
#SBATCH --time=160:00:00
#SBATCH --exclude=wadi-01,wadi-02,wadi-03,wadi-04,wadi-05
#SBATCH --mem-per-cpu=30g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH

  HF_DATASETS_CACHE=$PWD/.datasets_cache
  export HF_DATASETS_CACHE

  HF_HOME=$PWD/.hf_home
  export HF_HOME
}

activate
set_env_vars

module load cuda
module load nvidia

python3  src/zero_shot_experiment.py --config src/zero_shot_config.json
