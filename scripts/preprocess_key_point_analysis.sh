#!/bin/bash -x
#SBATCH --time=100:00:00
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

python3  src/point_analysis_preprocessing.py \
 --judges_results src/models_for_keypoint_analysis.json \
 --output_path point_analysis_preprocessing \
 --prompt 'zero-shot-good-speech-guidelines-short-cot' \
 --openai_api_key secret_keys/openai_key \
 --preprocessing_prompt 'prompts/preprocess_reasoning.txt' \
 --model_name gpt-4.1
