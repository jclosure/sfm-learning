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

## Repo auto-activate (direnv)

This repo has a committed `.envrc` that activates `datascience` automatically.

### One-time shell setup (zsh)

Add this to `~/.zshrc` if missing:

```bash
eval "$(direnv hook zsh)"
```

Then reload shell:

```bash
source ~/.zshrc
```

### One-time trust per repo

```bash
cd ~/projects/sfm-learning
direnv allow
```

After that, entering this directory auto-activates `datascience`, and leaving it restores your previous shell env.
