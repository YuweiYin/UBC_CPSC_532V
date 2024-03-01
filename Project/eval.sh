#!/bin/bash

echo -e "\n>>> Yuwei <<< Start job\n"

MODEL=$1
CACHE_DIR="/ubc/cs/research/nlp/yuweiyin/.cache/huggingface/"

run_eval(){
  task=$1
  echo -e "\n\n>>> Start of EVAL Model '${MODEL}' on Task '${task}'<<<\n"
  python3 eval.py --model "hf" \
    --model_args "pretrained=${MODEL},dtype=float" \
    --tasks "${task}" \
    --device "cuda:0" \
    --batch_size "auto:8" \
    --use_cache "${CACHE_DIR}" \
    --cache_requests "true" \
    --cache_dir "${CACHE_DIR}" \
    --seed 42 \
    --log_samples \
    --output_path "results/${task}---${MODEL}---eval"
  echo -e "\n>>> End of EVAL on ${task} <<<\n\n"
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
run_eval "super-glue-t5-prompt"

echo -e "\n\n>>> DONE EVAL <<<\n\n"
