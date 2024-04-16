#!/bin/bash

echo -e "\n>>> START EVAL --- RAG <<<\n"
conda activate 532v

CUDA=$1
MODEL=$2
DTYPE=$3  # "float32", "float16", "auto"
BSZ=$4  # "64", "32", "auto:8"
RAG=$5  # "wiki" "conceptNet" "arxiv" "googleSearch" "llm" "atomic"
ICL=$5  # 1, 3, 5, 8

CACHE_DIR="/ubc/cs/research/nlp/yuweiyin/.cache/huggingface/"
mkdir -p "${CACHE_DIR}"

run_eval(){
  task=$1
  echo -e "\n\n>>> Start of EVAL --- RAG <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${RAG}; ICL = ${ICL}\n"
  CUDA_VISIBLE_DEVICES=${CUDA} python3 eval.py --model "hf" \
    --model_args "pretrained=${MODEL},dtype=${DTYPE}" \
    --tasks "${task}" \
    --device "cuda" \
    --batch_size "${BSZ}" \
    --cache_dir "${CACHE_DIR}" \
    --seed 42 \
    --log_samples \
    --output_path "results_eval/${task}---rag-${RAG}" \
    --use_rag \
    --rag_source "${RAG}" \
    --rag_limit 10 \
    --llm_retriever_type "google" \
    --llm_agent_type "google" \
    --use_icl --icl_n_example "${ICL}"
  echo -e "\n>>> End of EVAL --- RAG <<< Model: ${MODEL}; Task: ${task}; RAG Source: ${RAG}; ICL = ${ICL}\n\n"
}

#for rag_source in "atomic" "llm_gemini" "llm_openai" "llm_anthropic" "wiki" "conceptNet" "arxiv" "googleSearch" "ALL"

run_eval "wsc273"
run_eval "winogrande"
#run_eval "anli"
run_eval "anli_r1"
run_eval "anli_r2"
run_eval "anli_r3"
#run_eval "ai2_arc"
run_eval "arc_easy"
run_eval "arc_challenge"
run_eval "piqa"
#run_eval "swag"  # TODO: Too large for RAG (skip for now) 20k
#run_eval "hellaswag"  # TODO: Too large for RAG (skip for now) 10k

#run_eval "glue"
run_eval "rte"
#run_eval "qnli"  # TODO: Too large for RAG (skip for now) 5.46k
#run_eval "mnli"  # TODO: Too large for RAG (skip for now) 9.8k
#run_eval "mnli_mismatch"  # TODO: Too large for RAG (skip for now) 9.85k
run_eval "mrpc"
#run_eval "qqp"  # TODO: Too large for RAG (skip for now) 391k
run_eval "wnli"
run_eval "sst2"
#run_eval "cola"  # TODO: Using MCC Metric (skip for now)

#run_eval "super-glue-lm-eval-v1"
run_eval "cb"
run_eval "wic"
run_eval "sglue_rte"
run_eval "boolq"
run_eval "copa"
#run_eval "multirc"  # TODO: Too large for RAG (skip for now) 9.69k
#run_eval "record"  # TODO: Too large for RAG (skip for now) 10k
run_eval "wsc"

echo -e "\n\n>>> DONE EVAL --- RAG <<<\n\n"
