# sfm-learning

A learning-focused **image-to-3D sparse reconstruction** project implementing a classic incremental SfM pipeline:

1. Feature extraction
2. Matching
3. Geometric verification
4. Incremental image registration
5. Triangulation
6. Outlier filtering (RANSAC stages)
7. Bundle adjustment (small global BA)

## Why this project

This is intentionally simple and readable. The goal is understanding the pipeline end-to-end, not replacing COLMAP.

## Project layout

- `src/sfm_learning/` core library
- `app.py` Streamlit web app
- `tests/` pytest unit tests
- `notebooks/` interactive notebook

## Quick start (conda)

```bash
cd ~/projects/sfm-learning
conda env create -f environment.yml
conda activate datascience
```

If the env already exists, refresh deps with:

```bash
cd ~/projects/sfm-learning
conda run -n datascience pip install -e '.[dev]'
```

## Auto-activation (direnv)

This repo includes a `.envrc` that auto-activates `datascience` when you `cd` here.

One-time machine setup:

```bash
brew install direnv
# ensure this exists in ~/.zshrc:
# eval "$(direnv hook zsh)"
source ~/.zshrc
```

One-time repo trust:

```bash
cd ~/projects/sfm-learning
direnv allow
```

## CLI usage

```bash
sfm-learning ./data/images -o outputs/sparse.ply
```

Options:
- `--detector sift|orb`
- `--no-ba`

## Web app

```bash
streamlit run app.py
```

Upload multiple overlapping photos and click **Run reconstruction**.

## Notebook

Open:

```bash
jupyter notebook notebooks/sfm_learning_walkthrough.ipynb
```

## Testing

```bash
pytest -q
```

## Notes on data capture

Good reconstructions need:
- 60-80% overlap between neighboring photos
- textured surfaces
- varied viewpoints (not just pure rotation)
- stable exposure and focus

## Known limitations

- Intrinsics are approximated from image size
- No dense MVS / meshing / texturing yet
- BA is intentionally lightweight

## Next improvements

- Load real intrinsics from EXIF/calibration
- Add track management and stronger outlier pruning
- Add dense reconstruction stage with OpenMVS/Open3D
- Add camera frustum visualization
