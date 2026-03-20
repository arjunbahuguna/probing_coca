# MusicCoCa Genre/Style Probing Results Summary

## Key Metrics

| Metric                | Value         | Pass Criteria         |
|---------------------- | ------------ | -------------------- |
| Prompt count          | 280          | == 280               |
| Embedding shape       | (280, 768)   | == (280, 768)        |
| Genre accuracy        | 1.000        | > 0.10 (chance)      |
| Style accuracy        | 0.567        | > 0.167 (chance)     |
| Subspace overlap      | 0.000        | —                    |
| Instrument sim shape  | (10, 10)     | == (10, 10)          |
| Instrument sim std    | 0.088        | > 0.01 (non-uniform) |

## Interpretation
- **Genre is linearly separable** in CoCa embedding (accuracy 100%).
- **Style is moderately separable** (accuracy 56.7%).
- **Genre and style subspaces are orthogonal** (overlap = 0.000).
- **Genre–instrument similarity matrix** shows meaningful variation (std = 0.088).

## Figures
- [Probe Accuracy Bar Chart (PDF)](outputs/figures/probe_accuracy.pdf)
- [Genre × Instrument Heatmap (PDF)](outputs/figures/genre_instrument_heatmap.pdf)
- [UMAP Genre/Style Visualization (PDF)](outputs/figures/umap_genre_style.pdf)

## Artifacts
- [metadata.json](outputs/metadata.json): Prompt taxonomy
- [embeddings.npy](outputs/embeddings.npy): Embedding matrix
- [probe_results.json](outputs/probe_results.json): Probe metrics
- [genre_instrument_similarity.npy](outputs/genre_instrument_similarity.npy): Similarity matrix

## Sanity Checks
- All required files and figures are present and validated.
- All metrics meet or exceed plan thresholds.

---
_Run `python scripts/validate_outputs.py` to re-check outputs automatically._
