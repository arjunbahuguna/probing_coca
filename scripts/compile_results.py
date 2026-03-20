"""Compile all results into a summary table."""
import json
import numpy as np

def main():
    with open("outputs/probe_results.json") as f:
        pr = json.load(f)
    sim = np.load("outputs/genre_instrument_similarity.npy")

    print("=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"\nGenre probe accuracy:  {pr['genre_accuracy_mean']:.1%} ± {pr['genre_accuracy_std']:.1%}")
    print(f"Style probe accuracy:  {pr['style_accuracy_mean']:.1%} ± {pr['style_accuracy_std']:.1%}")
    print(f"Subspace overlap (cos): {pr['subspace_overlap_cosine']:.3f}")
    print(f"\nInterpretation:")
    if pr['genre_accuracy_mean'] > 0.5:
        print("  → Genre is LINEARLY SEPARABLE in CoCa embedding")
    else:
        print("  → Genre is NOT linearly separable (negative result)")
    if abs(pr['subspace_overlap_cosine']) > 0.3:
        print(f"  → Genre and style subspaces are ENTANGLED (overlap={pr['subspace_overlap_cosine']:.3f})")
    else:
        print(f"  → Genre and style subspaces are ORTHOGONAL (overlap={pr['subspace_overlap_cosine']:.3f})")
    print(f"\nGenre–instrument similarity matrix shape: {sim.shape}")
    print(f"Max similarity: {sim.max():.3f}")
    print(f"Min similarity: {sim.min():.3f}")

if __name__ == "__main__":
    main()
