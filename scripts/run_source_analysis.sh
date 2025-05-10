#!/bin/bash -x
#SBATCH --time=1:00:00
#SBATCH -c1
#SBATCH --mem-per-cpu=1g

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

python3 src/source_analysis.py \
  --eval_models_config src/judges_results.json \
  --output_path source_analysis \
  --data_path data.csv \
  --prompt 'zero-shot-good-speech-guidelines' \
