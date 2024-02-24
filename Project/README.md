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

**Example**: Run the [GPT-2](https://huggingface.co/openai-community/gpt2) model (smallest: 124M parameters)
on the [Commonsense QA](https://huggingface.co/datasets/tau/commonsense_qa) dataset,
including fine-tuning, generation, and evaluation.

```bash
python3 main.py --ds_name "commonsense_qa" --model_name "gpt2" --verbose --eval_before --eval_after
```

To specify more hyperparameters (all the following settings are default values):

```bash
python3 main.py \
  --ds_name "commonsense_qa" \
  --model_name "gpt2" \
  --verbose \
  --eval_before \
  --eval_after \
  --seed 42 \
  --cuda "0" \
  --n_icl 5 \
  --n_gen 1 \
  --len_gen 10 \
  --epoch 5 \
  --bsz_train 32 \
  --bsz_gen 32 \
  --init_lr "1e-3" \
  --w_decay "5e-4" \
  --cache_dir "~/.cache/huggingface/datasets" \
  --log_dir "log" \
  --ckpt_dir "ckpt" \
  --output_dir "output"
```

## Experimental Results

- The running logs and all losses (`.log` file) will be in the folder `log/`
- The model checkpoints info (`.pt` file) after fine-tuning will be in the folder `ckpt/`
- The generation and evaluation results (`.jsonl` file) and statistics will be in the folder `output/`