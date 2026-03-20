"""UMAP visualization of CoCa embeddings."""
import json
import numpy as np
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os

def main():
    os.makedirs("outputs/figures", exist_ok=True)

    X = np.load("outputs/embeddings.npy")
    with open("outputs/metadata.json") as f:
        meta = json.load(f)

    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    X_2d = reducer.fit_transform(X)

    genre_labels = [m["genre_label"] for m in meta]
    unique_genres = sorted(set(genre_labels))
    genre_colors = {g: cm.tab10(i / len(unique_genres)) for i, g in enumerate(unique_genres)}

    style_labels = [m["style_label"] if m["style_label"] else "genre-only" for m in meta]
    unique_styles = sorted(set(style_labels))
    style_colors = {s: cm.Set2(i / len(unique_styles)) for i, s in enumerate(unique_styles)}

    set_labels = [m["set"] for m in meta]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: colored by genre
    for g in unique_genres:
        mask = [i for i, l in enumerate(genre_labels) if l == g]
        axes[0].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[genre_colors[g]], label=g, s=15, alpha=0.7)
    axes[0].set_title("Colored by Genre")
    axes[0].legend(fontsize=6, ncol=2)

    # Panel 2: colored by style
    for s in unique_styles:
        mask = [i for i, l in enumerate(style_labels) if l == s]
        axes[1].scatter(X_2d[mask, 0], X_2d[mask, 1], c=[style_colors[s]], label=s, s=15, alpha=0.7)
    axes[1].set_title("Colored by Style")
    axes[1].legend(fontsize=7)

    # Panel 3: colored by prompt set (genre-only vs joint)
    set_colors = {"genre_only": "#2196F3", "joint": "#FF9800"}
    for s in ["genre_only", "joint"]:
        mask = [i for i, l in enumerate(set_labels) if l == s]
        axes[2].scatter(X_2d[mask, 0], X_2d[mask, 1], c=set_colors[s], label=s, s=15, alpha=0.7)
    axes[2].set_title("Genre-only vs Style-modified")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])

    plt.suptitle("UMAP of MusicCoCa Text Embeddings (D=768 → 2D)", fontsize=14)
    plt.tight_layout()
    plt.savefig("outputs/figures/umap_genre_style.pdf", dpi=300)
    plt.savefig("outputs/figures/umap_genre_style.png", dpi=150)
    print("Saved UMAP → outputs/figures/umap_genre_style.pdf")

if __name__ == "__main__":
    main()
