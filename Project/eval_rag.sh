#!/bin/bash

echo -e "\n>>> START EVAL --- RAG <<<\n"
conda activate 532v

CUDA=$1
MODEL=$2
TASK=$3
RAG=$4
CACHE_DIR="/path/to/.cache/huggingface/"

run_eval(){
  task=$1
  rag=$2
  echo -e "\n\n>>> Start of EVAL --- RAG <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${rag}\n"
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
    --output_path "results/${task}---${MODEL}---eval_rag-${rag}" \
    --use_rag \
    --rag_source "${rag}"
  echo -e "\n>>> End of EVAL --- RAG <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${rag}\n\n"
}

#run_eval "wsc273"
#run_eval "winogrande"
#run_eval "anli"
#run_eval "ai2_arc"
#run_eval "piqa"
#run_eval "swag"
#run_eval "hellaswag"
#run_eval "glue"
#run_eval "super-glue-lm-eval-v1"

run_eval "${TASK}" "${RAG}"

echo -e "\n\n>>> DONE EVAL --- RAG <<<\n\n"
