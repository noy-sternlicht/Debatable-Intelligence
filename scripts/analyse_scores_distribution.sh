#!/bin/bash -x
#SBATCH --time=3:00:00
#SBATCH -c1
#SBATCH --mem=10g

PWD=$(pwd)
ENV_NAME="myenv"

activate() {
  . $PWD/myenv/bin/activate
}

set_env_vars() {
  PYTHONPATH=$PWD/src
  export PYTHONPATH
}

activate
set_env_vars

python3 src/analyse_scores_distribution.py \
  --eval_models_config src/agg_eval_models.json \
  --output_path scores_distribution_res \
  --prompt 'zero-shot-good-speech-guidelines' \
