#!/bin/bash -x
#SBATCH --time=40:00:00
#SBATCH --exclude=wadi-01,wadi-02,wadi-03,wadi-04,wadi-05
#SBATCH --mem-per-cpu=20g

echo $PWD
activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH
}

activate
set_env_vars

python3 src/generate_speeches.py --config src/generate_speeches_config.json
