# UBC CPSC 532V (2023W2) Assignment 2

- [Report](./UBC_CPSC_532V-A2-Report.pdf)
- [Jupyter Notebook](./UBC_CPSC_532V-A2.ipynb)

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

## Run the code

```bash
jupyter notebook
```

Run `UBC_CPSC_532V-A2.ipynb`

## Experimental Results

See `UBC_CPSC_532V-A2.ipynb` and the folder `output/`
