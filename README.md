# probing_coca

Linear probing experiments on multimodal MusicCoCa embeddings to test separability of musical concepts from text prompts and MTAT audio clips

## What this repository does

This project evaluates whether MusicCoCa embedding space linearly separates:

- Genre and style (core probe)
- Genre-to-instrument semantic alignment (probe weight correlation)
- Additional musical attributes (tempo, timbre, structure, melody, octave, duration, loudness, instrument)
- Text-vs-audio separability gaps using a quick MTAT protocol

It produces reproducible artifacts in `outputs/` and markdown summaries in the repository root.

## Repository structure

- `scripts/`: data prep, embedding extraction, probing, visualization, and validation scripts
- `outputs/`: generated artifacts (`.npy`, `.json`, and figures)
- `results_summary.md`: core genre/style findings
- `results_additional_attributes.md`: additional text-attribute probe findings
- `results_text_vs_audio.md`: quick text-vs-audio comparison findings

## Prerequisites

### 1) Environment

- Python 3.10+ recommended
- A local clone of `magenta-realtime` available as a sibling directory
- Dependencies used by this repo:
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn`
  - `umap-learn`

### 2) Expected relative layout

The scripts import MusicCoCa by adding `../../magenta-realtime` to `sys.path`, so this repository should be placed so that both directories are siblings:

```text
<workspace>/
  magenta-realtime/
  probing_coca/
```

### 3) Optional MTAT data (for audio branch)

Place MTAT under `probing_coca/mtat/` with:

- `annotations_final.csv`
- `clip_info_final.csv`
- clip audio files at the paths referenced by `mp3_path`

## Installation

From this repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scikit-learn matplotlib seaborn umap-learn
```

If your local `magenta-realtime` setup requires extra dependencies, install them in the same environment before running embedding scripts.

## Text-only pipeline (core)

Run from the repository root in this order:

```bash
python scripts/generate_prompts.py
python scripts/extract_embeddings.py
python scripts/probe_genre_style.py
python scripts/visualize_umap.py
python scripts/instrument_correlation.py
python scripts/compile_results.py
python scripts/validate_outputs.py
```

### Additional text attributes

```bash
python scripts/probe_additional_attributes.py
```

## Optional MTAT audio branch (quick run)

```bash
python scripts/prepare_mtat_tasks.py \
  --mtat-root mtat \
  --annotations mtat/annotations_final.csv \
  --clip-info mtat/clip_info_final.csv

python scripts/extract_audio_embeddings_mtat.py \
  --tasks outputs/mtat_tasks.json \
  --mtat-root mtat

python scripts/probe_mtat_audio_tags.py \
  --tasks outputs/mtat_tasks.json \
  --embeddings outputs/mtat_audio_embeddings.npy \
  --metadata outputs/mtat_audio_metadata.json

python scripts/compare_text_vs_audio.py \
  --audio-results outputs/mtat_audio_probe_results.json \
  --text-results outputs/additional_probe_results.json
```

## Main outputs

Key artifacts generated in `outputs/`:

- `metadata.json`: prompt taxonomy
- `embeddings.npy`: text embedding matrix
- `probe_results.json`: genre/style probe metrics
- `W_genre.npy`, `W_style.npy`: learned linear probe weight matrices
- `pairwise_overlap.npy`: genre-vs-style per-class cosine overlap
- `genre_instrument_similarity.npy`: genre probe weights vs instrument text embeddings
- `additional_probe_results.json`: additional attribute probe metrics
- `mtat_tasks.json`: MTAT quick-run task definitions
- `mtat_audio_embeddings.npy`: MTAT audio embedding matrix
- `mtat_audio_probe_results.json`: audio probe metrics
- `text_vs_audio_comparison.json`: merged text-vs-audio table

Figures are written to `outputs/figures/` as `.png` and/or `.pdf`.

## Reported results snapshot

From current committed summaries:

- Genre probe accuracy: `1.000` (10-class)
- Style probe accuracy: `0.567` (6-class)
- Subspace overlap cosine (genre vs style mean directions): `0.000`
- Additional text-attribute probes: all listed attributes above chance, with strongest results for instrument/structure/chord-choice
- Quick MTAT comparison: audio can outperform text for some proxies (e.g., loudness), while text remains stronger on others (e.g., instrument in this setup)

See detailed tables in:

- `results_summary.md`
- `results_additional_attributes.md`
- `results_text_vs_audio.md`

## Reproducibility notes

- Cross-validation is stratified and uses fixed random states where specified in scripts.
- UMAP uses a fixed random seed.
- MTAT quick run enforces artist-disjoint folds via grouped splitting.
- First use of MusicCoCa may trigger model asset downloads.

## License

MIT. See `LICENSE`.