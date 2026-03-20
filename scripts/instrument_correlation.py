"""Correlate genre probe weights with instrument embeddings."""
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os

# Add magenta-realtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "magenta-realtime"))

from magenta_rt import musiccoca

INSTRUMENTS = [
    "kick drum", "synthesizer", "piano", "guitar", "bass",
    "violin", "trumpet", "hi-hat", "saxophone", "pad",
]

def main():
    # Load genre probe weights
    W_genre = np.load("outputs/W_genre.npy")  # [n_genres, 768]
    with open("outputs/probe_results.json") as f:
        results = json.load(f)
    genre_classes = results["genre_classes"]

    # Get instrument embeddings from MusicCoCa
    model = musiccoca.MusicCoCa()
    instr_embeddings = model.embed_batch_text(INSTRUMENTS)  # [10, 768]

    # Compute genre × instrument similarity matrix
    sim_matrix = cos_sim(W_genre, instr_embeddings)  # [n_genres, n_instruments]

    # Save
    np.save("outputs/genre_instrument_similarity.npy", sim_matrix)

    # --- Figure: Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(
        sim_matrix,
        xticklabels=INSTRUMENTS,
        yticklabels=genre_classes,
        annot=True, fmt=".2f",
        cmap="RdBu_r", center=0,
        ax=ax,
    )
    ax.set_title("Genre Probe Weight × Instrument Embedding Cosine Similarity")
    ax.set_xlabel("Instrument")
    ax.set_ylabel("Genre")
    plt.tight_layout()
    plt.savefig("outputs/figures/genre_instrument_heatmap.pdf", dpi=300)
    plt.savefig("outputs/figures/genre_instrument_heatmap.png", dpi=150)
    print("Saved heatmap → outputs/figures/genre_instrument_heatmap.pdf")

if __name__ == "__main__":
    main()
