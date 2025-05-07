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

python3 src/analyse_pairwise_agreement.py \
  --eval_models_config src/judges_results.json \
  --data_path data.csv \
  --output_dir pairwise_agreement_results \
  --min_shared_annotations 50 \
  --prompt 'zero-shot-good-speech-guidelines'
