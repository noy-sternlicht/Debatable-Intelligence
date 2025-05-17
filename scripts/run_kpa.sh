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

python3 src/key_point_analysis.py \
  --preprocessed_judge_results data_for_keypoint_analysis.csv \
  --keypoints 'keypoints.csv' \
  --output_path 'keypoint_analysis' \
  --judge Llama-3.3-70B
