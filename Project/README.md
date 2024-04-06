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

* For generation/evaluation only:

```bash
pip install -r requirements_eval.txt
```

* For training/fine-tuning:

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

## Experimental Results

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
#pip install vllm
```

Command line tool `lm-eval` usage (basic example):

```bash
lm_eval --model "hf" \
  --model_args "pretrained=EleutherAI/pythia-160m,revision=step100000,dtype=auto" \
  --tasks "lambada_openai,hellaswag" \
  --device "cuda:0" \
  --batch_size "auto:4" \
  --use_cache "/path/to/.cache/huggingface/" \
  --cache_requests "true" \
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
  --use_cache "/path/to/.cache/huggingface/" \
  --cache_requests "true" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "eval_results/copa---gpt2"
```

### Evaluate the Fine-tuned Models

Suppose the fine-tuned checkpoint under the `"runs/commonsense_qa---gpt2/ckpt/model_epoch_9/"` folder:

```bash
python3 eval.py --model "hf" \
  --model_args "pretrained=runs/commonsense_qa---gpt2/ckpt/model_epoch_9,dtype=auto" \
  --tasks "copa" \
  --device "cuda:0" \
  --batch_size "auto:8" \
  --use_cache "/path/to/.cache/huggingface/" \
  --cache_requests "true" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "eval_results/copa---gpt2_ft"
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
  --use_cache "/path/to/.cache/huggingface/" \
  --cache_requests "true" \
  --cache_dir "/path/to/.cache/huggingface/" \
  --seed 42 \
  --log_samples \
  --output_path "eval_results/copa---gpt2" \
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

**Multilingual** (TBD)

* XCOPA: `xcopa`
* XNLI: `xnli`

### Baselines

* `openai-community/gpt2`: The smallest version of GPT-2, with 124M parameters. ([LINK](https://huggingface.co/openai-community/gpt2))

### Experimental Results

* Before running `eval.sh` or `eval_rag.sh`, please modify the `CACHE_DIR` variable.

```bash
bash eval.sh "0" "openai-community/gpt2"
```

| Task                   | Version | Filter | n-shot | Metric | GPT-2 (Small)    |
|------------------------|---------|--------|--------|--------|------------------|
| **GLUE**               | N/A     | None   | None   | mcc    | 0.0126 (±0.0315) |
| **GLUE**               |         |        |        | acc    | 0.4658 (±0.0019) |
| **GLUE**               |         |        |        | f1     | 0.3786 (±0.0035) |
| GLUE - `rte`           | 1       | None   | None   | acc    | 0.5307 (±0.0300) |
| GLUE - `qnli`          | 1       | None   | None   | acc    | 0.5017 (±0.0068) |
| GLUE - `mnli`          | 1       | None   | None   | acc    | 0.3372 (±0.0048) |
| GLUE - `mnli_mismatch` | 1       | None   | None   | acc    | 0.3321 (±0.0047) |
| GLUE - `mrpc`          | 1       | None   | None   | acc    | 0.5613 (±0.0246) |
| GLUE - `mrpc`          |         | None   | None   | f1     | 0.6832 (±0.0226) |
| GLUE - `qqp`           | 1       | None   | None   | acc    | 0.5215 (±0.0025) |
| GLUE - `qqp`           |         | None   | None   | f1     | 0.3755 (±0.0035) |
| GLUE - `wnli`          | 1       | None   | None   | acc    | 0.4225 (±0.0590) |
| GLUE - `sst2`          | 1       | None   | None   | acc    | 0.5505 (±0.0169) |
| GLUE - `cola`          | 1       | None   | None   | mcc    | 0.0126 (±0.0315) |


| Task                    | Version | Filter | n-shot | Metric | GPT-2 (Small)    |
|-------------------------|---------|--------|--------|--------|------------------|
| **SuperGLUE**           | N/A     | None   | None   | acc    | 0.5116 (±0.0052) |
| **SuperGLUE**           |         | None   | None   | em     | 0.2573 (±0.0044) |
| **SuperGLUE**           |         | None   | None   | f1     | 0.2649 (±N/A)    |
| SuperGLUE - `cb`        | 1       | None   | None   | acc    | 0.4107 (±0.0663) |
| SuperGLUE - `cb`        |         | None   | None   | f1     | 0.2619 (±N/A)    |
| SuperGLUE - `wic`       | 1       | None   | None   | acc    | 0.4922 (±0.0198) |
| SuperGLUE - `sglue_rte` | 0       | None   | None   | acc    | 0.5307 (±0.0300) |
| SuperGLUE - `boolq`     | 2       | None   | None   | acc    | 0.4872 (±0.0087) |
| SuperGLUE - `copa`      | 1       | None   | None   | acc    | 0.6200 (±0.0488) |
| SuperGLUE - `multirc`   | 1       | None   | None   | acc    | 0.5301 (±0.0072) |
| SuperGLUE - `record`    | 1       | None   | None   | f1     | 0.2649 (±0.0044) |
| SuperGLUE - `record`    |         | None   | None   | em     | 0.2573 (±0.0044) |
| SuperGLUE - `wsc`       | 1       | None   | None   | acc    | 0.4327 (±0.0488) |


| Task                        | Version | Filter | n-shot | Metric | GPT-2 (Small)    |
|-----------------------------|---------|--------|--------|--------|------------------|
| **WSC273** `wsc273`         | 1       | None   | None   | acc    | 0.5861 (±0.0299) |
| **WinoGrande** `winogrande` | 1       | None   | None   | acc    | 0.5162 (±0.0140) |
| **ANLI** `anli`             | N/A     | None   | None   | acc    | 0.3434 (±0.0084) |
| ANLI - `anli_r1`            | 1       | None   | None   | acc    | 0.3410 (±0.0150) |
| ANLI - `anli_r2`            | 1       | None   | None   | acc    | 0.3390 (±0.0150) |
| ANLI - `anli_r3`            | 1       | None   | None   | acc    | 0.3492 (±0.0138) |
| **ARC** `ai2_arc`           | N/A     | None   | None   | acc    | 0.3563 (±0.0078) |
| ARC - `arc_easy`            | 1       | None   | None   | acc    | 0.4381 (±0.0102) |
| ARC - `arc_challenge`       | 1       | None   | None   | acc    | 0.1903 (±0.0115) |
| **PIQA** `piqa`             | 1       | None   | None   | acc    | 0.6289 (±0.0113) |
| **SWAG** `swag`             | 1       | None   | None   | acc    | 0.4052 (±0.0035) |
| **HellaSwag** `hellaswag`   | 1       | None   | None   | acc    | 0.2892 (±0.0045) |

---
