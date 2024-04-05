#!/bin/bash

echo -e "\n>>> START EVAL <<<\n"
conda activate 532v

CUDA=$1
MODEL=$2
CACHE_DIR="/path/to/.cache/huggingface/"

run_eval(){
  task=$1
  echo -e "\n\n>>> Start of EVAL Model '${MODEL}' on Task '${task}'<<<\n"
  CUDA_VISIBLE_DEVICES=${CUDA} python3 eval.py --model "hf" \
    --model_args "pretrained=${MODEL},dtype=float" \
    --tasks "${task}" \
    --device "cuda" \
    --batch_size "auto:8" \
    --use_cache "${CACHE_DIR}" \
    --cache_requests "true" \
    --cache_dir "${CACHE_DIR}" \
    --seed 42 \
    --log_samples \
    --output_path "results/${task}---${MODEL}---eval"
  echo -e "\n>>> End of EVAL <<< Model: ${MODEL}; Task: ${task}\n\n"
}

run_eval "wsc273"
run_eval "winogrande"
run_eval "anli"
run_eval "ai2_arc"
run_eval "piqa"
run_eval "swag"
run_eval "hellaswag"
run_eval "glue"
run_eval "super-glue-lm-eval-v1"

echo -e "\n\n>>> DONE EVAL <<<\n\n"
