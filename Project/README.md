# UBC CPSC 532V (2023W2) Project

* [Project Proposal Document](./docs/Project_Proposal_Document.pdf)
* [Project Proposal Slides](./docs/Project_Proposal_Slides.pdf)

## Environment (Linux; macOS)

### Miniconda3

```bash
# https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
#curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh

~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

Then, reopen the terminal.

### Python3 Virtual Environments

Now, create the conda venv (do not use the most-updated Python3 version):

```bash
conda create -n 532v -y python=3.10
conda activate 532v
```

### Python3 Packages

```bash
git clone https://github.com/YuweiYin/UBC_CPSC_532V
cd Project/
```

* For generation/evaluation only:

```bash
pip install -r requirements_eval.txt
```

* For training/fine-tuning (optional):

```bash
pip install -r requirements_train.txt
pip install accelerate -U
pip install deepspeed
#pip install vllm
#pip install mamba-ssm
#pip install causal-conv1d
#pip install promptsource
```

## Experiments

### Run the code (on a CPU or a single GPU)

**Example**: Run the [GPT-2](https://huggingface.co/openai-community/gpt2) model (smallest: 124M parameters)
on the [Commonsense QA](https://huggingface.co/datasets/tau/commonsense_qa) dataset,
including training (fine-tuning), generation, and evaluation.

```bash
python3 train.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5 --use_lr_scheduler
```

Without evaluation:

```bash
python3 train.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0" --verbose \
  --save_after_epoch --ckpt_limit 5 --use_lr_scheduler
```

To show the training/evaluation logs and save logs to [wandb](https://wandb.ai/):

```bash
python3 train.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0" --verbose \
  --save_after_epoch --ckpt_limit 5 --use_lr_scheduler --use_wandb
```

To specify more hyperparameters (all the following settings are default values):

```bash
python3 train.py \
  --ds_name "commonsense_qa" \
  --model_name "gpt2" \
  --cuda "0" \
  --seed 42 \
  --verbose \
  --eval_before \
  --eval_after \
  --do_eval_epoch \
  --do_eval_batch \
  --save_after_epoch \
  --ckpt_limit 5 \
  --eval_gap 1000 \
  --logging_gap 100 \
  --n_icl 5 \
  --n_gen 1 \
  --len_gen 10 \
  --epoch 5 \
  --bsz_train 32 \
  --bsz_gen 32 \
  --init_lr "1e-3" \
  --use_lr_scheduler \
  --w_decay "5e-4" \
  --save_dir "" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --log_dir "log" \
  --ckpt_dir "ckpt" \
  --output_dir "output"
```

### Run the code (on multiple GPUs)

**DataParallel** (`torch.nn.parallel.data_parallel.DataParallel`) Training (at least two GPUs):

```bash
python3 train_dp.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0,1" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5 --use_lr_scheduler \
  --dp
```

**DistributedDataParallel** (`torch.nn.parallel.distributed.DistributedDataParallel`) Training (at least two GPUs):

```bash
python3 train_ddp.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0,1" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5 --use_lr_scheduler \
  --backend "gloo" --master_addr "localhost" --master_port "12345" --ddp --ddp_gen
```

* Add `--verbose_all` if you want all GPU devices to show logs.
* Remove `--ddp_gen` to run generation on one GPU device (rank=0). (slower when the valid & test set is large.)

**Model Parallel** ([PyTorch](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html), 
[Transformers](https://huggingface.co/transformers/v4.9.2/parallelism.html)) is also recommended.

## Experiments

### Training Results

- The running logs and all training losses will be in the folder `f"runs/{save_dir}/log/"`
- The model checkpoints with tokenizers info after training will be in the folder `f"runs/{save_dir}/ckpt/"`
- The generation and evaluation results during training will be in the folder `f"runs/{save_dir}/output/"`
- The final generation and evaluation results with statistics will be in the folder `f"/results/"`

### Generation and Evaluation Method

Use [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

Command line tool `lm-eval` usage (basic example):

```bash
lm_eval --model "hf" \
  --model_args "pretrained=EleutherAI/pythia-160m,revision=step100000,dtype=auto" \
  --tasks "lambada_openai,hellaswag" \
  --device "cuda:0" \
  --batch_size "auto:4" \
  --seed 42
```

* [Basic Usage](https://github.com/EleutherAI/lm-evaluation-harness?tab=readme-ov-file#basic-usage);
* [Command-line Interface](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md);
* [Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html);
* [Supported Tasks](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks)
  * Or try `lm-eval --tasks list`

We clone the [folder](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval) to
the local [module](./lm_eval/) `./lm_eval`, add some new features, and use it as our evaluation codebase.
For example, we support passing the `cache_dir` parameter when loading datasets, models, and tokenizers.

Hence, **we recommend using the following evaluation script**:

```bash
python3 eval.py --model "hf" \
  --model_args "pretrained=gpt2,dtype=auto" \
  --tasks "copa" \
  --device "cuda:0" \
  --batch_size "auto:8" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "results_eval/copa---gpt2"
```

### Evaluate the Fine-tuned Models

Suppose the fine-tuned checkpoint under the `"runs/commonsense_qa---gpt2/ckpt/model_epoch_9/"` folder:

```bash
python3 eval.py --model "hf" \
  --model_args "pretrained=runs/commonsense_qa---gpt2/ckpt/model_epoch_9,dtype=auto" \
  --tasks "copa" \
  --device "cuda:0" \
  --batch_size "auto:8" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "results_eval/copa---gpt2_ft"
```

### Evaluation with RAG

* `--rag_source` can be: `"atomic"`, `"llm_gemini"`, `"llm_openai"`, `"llm_anthropic"`, `"wiki"`, `"conceptNet"`, `"arxiv"`, `"googleSearch"` or `"ALL"`. Default value `"ALL"` means use all RAG sources.
* Before using LLMs (`"llm_gemini"`, `"llm_openai"`, or `"llm_anthropic"`), please specify the API keys in `rag/api_setup.py`

```bash
python3 eval.py --model "hf" \
  --model_args "pretrained=gpt2,dtype=auto" \
  --tasks "copa" \
  --device "cuda:0" \
  --batch_size "auto:8" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "results_eval/copa---gpt2" \
  --use_rag \
  --rag_source "ALL" \
```

### Testing Tasks and Datasets 

We choose the following datasets to perform evaluation.

* GLUE: `glue`
  * including `cola`, `mnli`, `mrpc`, `qnli`, `qqp`, `rte`, `sst2`, and `wnli`.
* SuperGLUE: `super-glue-lm-eval-v1` (or `super-glue-t5-prompt`)
  * including `boolq`, `cb`, `copa`, `multirc`, `record`, `rte`, `wic`, and `wsc`
* WSC273: `wsc273`
* WinoGrande: `winogrande`
* ANLI: `anli` (Evaluates `anli_r1`, `anli_r2`, and `anli_r3`)
* ARC: `ai2_arc` (Evaluates `arc_easy` and `arc_challenge`)
* PIQA: `piqa`
* SWAG: `swag`
* HellaSwag: `hellaswag`

### Baselines

* `openai-community/openai-gpt`: GPT-1
* `openai-community/gpt2`: The smallest version of GPT-2, with 124M parameters. ([LINK](https://huggingface.co/openai-community/gpt2))
* `openai-community/gpt2-medium`: 
* `openai-community/gpt2-large`: 
* `openai-community/gpt2-xl`: 


### Experimental Results

* Before running `eval.sh` or `eval_rag.sh`, please modify the `CACHE_DIR` variable (`""` for default Hugging Face directory: `"~/.cache/huggingface/"`).

**Without RAG**:

```bash
bash eval_all.sh "0" "openai-community/gpt2" "float16" "auto:8"

#mkdir -p log
#nohup bash eval_all.sh "0" "openai-community/gpt2" "float16" "auto:8" > log/eval---gpt2---all.log 2>&1 &
```

**With RAG**:

```bash
bash eval_all_rag.sh "0" "openai-community/gpt2" "float16" "auto:8" "wiki"

#mkdir -p log
#nohup bash eval_all_rag.sh "0" "openai-community/gpt2" "float16" "auto:8" "wiki" > log/eval---gpt2---all---rag-wiki.log 2>&1 &
```

---
