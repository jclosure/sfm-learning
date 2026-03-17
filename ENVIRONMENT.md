# Environment setup

This project is standardized on the conda environment:

- **Name:** `datascience`
- **Python:** 3.11
- **Install mode:** editable package + dev extras (`-e '.[dev]'`)

## One-time setup

```bash
cd ~/projects/sfm-learning
conda env create -f environment.yml
conda activate datascience
```

## Update dependencies

```bash
cd ~/projects/sfm-learning
conda run -n datascience pip install -e '.[dev]'
```

## Verify

```bash
conda run -n datascience python -V
conda run -n datascience pytest -q
```
