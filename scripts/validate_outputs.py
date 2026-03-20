"""
Validation script for MusicCoCa genre/style probing pipeline.
Checks all required outputs, shapes, and key metrics.
"""
import json
import numpy as np
from pathlib import Path

base = Path('outputs')
figs = [
    'probe_accuracy.pdf','probe_accuracy.png',
    'genre_instrument_heatmap.pdf','genre_instrument_heatmap.png',
    'umap_genre_style.pdf','umap_genre_style.png'
]

results = {}

# Metadata
with open(base/'metadata.json') as f:
    meta = json.load(f)
results['metadata_count'] = len(meta)

# Embeddings
X = np.load(base/'embeddings.npy')
results['embeddings_shape'] = X.shape

# Probe results
with open(base/'probe_results.json') as f:
    pr = json.load(f)
results['genre_acc'] = pr['genre_accuracy_mean']
results['style_acc'] = pr['style_accuracy_mean']
results['genre_chance_pass'] = pr['genre_accuracy_mean'] > 0.10
results['style_chance_pass'] = pr['style_accuracy_mean'] > (1/6)
results['subspace_overlap'] = pr['subspace_overlap_cosine']

# Weights
results['W_genre_shape'] = np.load(base/'W_genre.npy').shape
results['W_style_shape'] = np.load(base/'W_style.npy').shape
results['pairwise_overlap_shape'] = np.load(base/'pairwise_overlap.npy').shape

# Instrument similarity
sim = np.load(base/'genre_instrument_similarity.npy')
results['sim_shape'] = sim.shape
results['sim_std'] = float(sim.std())
results['sim_max'] = float(sim.max())
results['sim_min'] = float(sim.min())

# Figures
results['figures'] = {}
for f in figs:
    p = base/'figures'/f
    results['figures'][f] = {'exists': p.exists(), 'size': p.stat().st_size if p.exists() else -1}

# Print summary
print("Validation Results:")
for k, v in results.items():
    print(f"{k}: {v}")

# Sanity checks
assert results['metadata_count'] == 280, "Prompt count mismatch"
assert results['embeddings_shape'] == (280, 768), "Embedding shape mismatch"
assert results['genre_chance_pass'], "Genre accuracy below chance"
assert results['style_chance_pass'], "Style accuracy below chance"
assert results['sim_shape'] == (10, 10), "Instrument similarity shape mismatch"
for f, info in results['figures'].items():
    assert info['exists'] and info['size'] > 1000, f"Figure {f} missing or too small"
print("All checks passed.")
