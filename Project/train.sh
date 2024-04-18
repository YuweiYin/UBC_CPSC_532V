#!/bin/bash

echo -e "\n>>> START TRAINING <<<\n"
conda activate 532v

CUDA=$1
TASK=$2
MODEL=$3
BSZ=$4  # default: "32"
EPOCH=$5  # default: "5"
CKPT_LIMIT=$6  # default: "5"
N_ICL=$7  # default: "5"

CACHE_DIR="/ubc/cs/research/nlp/yuweiyin/.cache/huggingface/"
mkdir -p "${CACHE_DIR}"

run_train(){
  task=$1
  echo -e "\n>>> Start TRAINING <<< Task: ${task}; Model: ${MODEL}\n"
  CUDA_VISIBLE_DEVICES=${CUDA} python3 train.py \
  --cuda "0" --verbose --seed 42 \
  --ds_name "${task}" \
  --model_name "${MODEL}" \
  --eval_gap 1000 --logging_gap 100 \
  --n_icl "${N_ICL}" \
  --n_gen 1 --len_gen 10 \
  --epoch "${EPOCH}" --save_after_epoch --ckpt_limit "${CKPT_LIMIT}" \
  --bsz_train "${BSZ}" --bsz_gen "${BSZ}" \
  --init_lr "1e-3" --use_lr_scheduler --w_decay "5e-4" \
  --save_dir "${task}---${MODEL}---bsz${BSZ}_epoch${EPOCH}_cuda${CUDA}" \
  --cache_dir "${CACHE_DIR}" \
  --log_dir "log" --ckpt_dir "ckpt" --output_dir "output"
# --eval_before --eval_after --do_eval_epoch --do_eval_batch
  echo -e "\n>>> End TRAINING <<< Task: ${task}; Model: ${MODEL}\n"
}

run_train "${TASK}"

echo -e "\n\n>>> DONE TRAINING <<<\n\n"
