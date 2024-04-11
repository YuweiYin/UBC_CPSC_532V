#!/bin/bash

echo -e "\n>>> START EVAL --- RAG with Post-preprocessing <<<\n"
conda activate 532v

CUDA=$1
MODEL=$2
DTYPE=$3  # "float32", "float16", "auto"
BSZ=$4  # "64", "32", "auto:8"
TASK=$5
RAG=$6  # "wiki" "conceptNet" "arxiv" "googleSearch" "llm" "atomic"
RAG_POST=$7  # "ranking_documents" "summarizing_documents" "extracting_key_info"
# "refining_documents" "evaluating_documents" "identifying_conflict" "filter_duplication" "structured_format"

CACHE_DIR="/ubc/cs/research/nlp/yuweiyin/.cache/huggingface/"
mkdir -p "${CACHE_DIR}"

run_eval(){
  task=$1
  echo -e "\n\n>>> Start of EVAL --- RAG with Post-preprocessing <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${RAG}; RAG Post-processing Type: ${RAG_POST}\n"
  CUDA_VISIBLE_DEVICES=${CUDA} python3 eval.py --model "hf" \
    --model_args "pretrained=${MODEL},dtype=${DTYPE}" \
    --tasks "${task}" \
    --device "cuda" \
    --batch_size "${BSZ}" \
    --cache_dir "${CACHE_DIR}" \
    --seed 42 \
    --log_samples \
    --output_path "results_eval/${task}---rag-${RAG}---post-${RAG_POST}" \
    --use_rag \
    --rag_source "${RAG}" \
    --rag_limit 10 \
    --llm_retriever_type "google" \
    --llm_agent_type "google" \
    --use_rag_postprocess \
    --rag_postprocess_type "${RAG_POST}"
  echo -e "\n>>> End of EVAL --- RAG with Post-preprocessing <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${RAG}\n\n"
}

run_eval "${TASK}"

echo -e "\n\n>>> DONE EVAL --- RAG with Post-preprocessing <<<\n\n"
