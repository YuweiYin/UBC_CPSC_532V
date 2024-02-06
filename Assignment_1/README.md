# UBC CPSC 532V (2023W2) Assignment 1

- [Report](./UBC_CPSC_532V-A1-Report.pdf)

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

## Run the code

```bash
python3 main.py
```

The default setting is to run on all examples (from 1 to 10) with a `max_depth` of 6.

If you want to run only on one example (say, 5), 
specify the example id by passing the `-e` or `--example_id` parameter as follows:

```bash
python3 main.py -e 5
```

With a `max_depth` of 6, the Dijkstra Path Search procedure usually takes about 90-180 seconds.
If you want to change `max_depth` (say, 8), 
specify the depth by passing the `-d` or `--max_depth` parameter as follows:

```bash
python3 main.py -d 8
```

If you want to use LLM (like ChatGPT) to transform the (`source_word`, `relation`, `target_word`) pair into
a natural language sentence, please set your `"OPENAI_API_KEY"` in the global environment variables or
specify the key in the `__init__` method of `TextConverter` class in the `TextConverter.py` file,
and then pass the `-t` or `--use_gpt` parameter as follows:

```bash
python3 main.py -t
```

## Experiments

```bash
mkdir log
mkdir output
mkdir output_text
mkdir output_figure
```

- The running logs (`.log` file) will be in the folder `log/`
- The QA evaluation results (`.jsonl` file) and statistics will be in the folder `output/`
- The meta info (`.json` file) of each valid path will be in the folder `output_text/`
- The visualization figure (`.pdf` file) of each valid path will be in the folder `output_figure/`

```bash
python3 main.py -d 6
python3 main.py -d 8 -e 6  # For the 6th example, `max_depth = 6` will result in no valid path for any choice.
```
