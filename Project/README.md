# UBC CPSC 532V (2023W2) Project

## Environment (Linux; macOS)

### Miniconda3

```bash
# https://docs.conda.io/projects/miniconda/en/latest/
mkdir -p ~/miniconda3
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh -o ~/miniconda3/miniconda.sh
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
pip install -r requirements.txt
```

```bash
pip install accelerate -U
```

```bash
# https://spacy.io/usage
# pip install -U pip setuptools wheel
pip install 'spacy[apple]'
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
```

For macOS, `pip install graphviz` is not enough, run either of the following scripts:

```bash
brew graphviz
conda install graphviz
```

## Experiments

```bash
mkdir -p log
mkdir -p ckpt
mkdir -p output
```

### Run the code (on a CPU or a single GPU)

**Example**: Run the [GPT-2](https://huggingface.co/openai-community/gpt2) model (smallest: 124M parameters)
on the [Commonsense QA](https://huggingface.co/datasets/tau/commonsense_qa) dataset,
including training (fine-tuning), generation, and evaluation.

```bash
python3 main.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5
```

Without evaluation:

```bash
python3 main.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0" --verbose \
  --save_after_epoch --ckpt_limit 5
```

To specify more hyperparameters (all the following settings are default values):

```bash
python3 main.py \
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
  --w_decay "5e-4" \
  --save_dir "" \
  --cache_dir "~/.cache/huggingface/" \
  --log_dir "log" \
  --ckpt_dir "ckpt" \
  --output_dir "output"
```

### Run the code (on multiple GPUs)

**DataParallel** (`torch.nn.parallel.data_parallel.DataParallel`) Training (at least two GPUs):

```bash
python3 main_dp.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0,1" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5 \
  --dp
```

**DistributedDataParallel** (`torch.nn.parallel.distributed.DistributedDataParallel`) Training (at least two GPUs):

```bash
python3 main_ddp.py --ds_name "commonsense_qa" --model_name "gpt2" --cuda "0,1" --verbose \
  --eval_before --eval_after --do_eval_epoch --do_eval_batch --save_after_epoch --ckpt_limit 5 \
  --backend "gloo" --master_addr "localhost" --master_port "12345" --ddp
```

**Model Parallel** ([PyTorch](https://pytorch.org/tutorials/intermediate/model_parallel_tutorial.html), 
[Transformers](https://huggingface.co/transformers/v4.9.2/parallelism.html)) is also recommended.

## Experimental Results

- The running logs and all training losses will be in the folder `log/`
- The model checkpoints with tokenizers info after training will be in the folder `ckpt/`
- The generation and evaluation results with statistics will be in the folder `output/`

---
