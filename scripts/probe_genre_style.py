"""Linear probing for genre and style classification."""
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

def main():
    X = np.load("outputs/embeddings.npy")       # [N, 768]
    with open("outputs/metadata.json") as f:
        meta = json.load(f)

    os.makedirs("outputs/figures", exist_ok=True)

    # --- Probe 1: Genre classification (all 280 prompts) ---
    genre_labels = [m["genre_label"] for m in meta]
    le_genre = LabelEncoder()
    y_genre = le_genre.fit_transform(genre_labels)

    probe_genre = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    genre_scores = cross_val_score(probe_genre, X, y_genre, cv=cv, scoring="accuracy")

    # Fit on full data to extract weight vectors
    probe_genre.fit(X, y_genre)
    W_genre = probe_genre.coef_  # [n_genres, 768]

    # --- Probe 2: Style classification (joint set only, 180 prompts) ---
    joint_mask = [i for i, m in enumerate(meta) if m["style_label"] is not None]
    X_joint = X[joint_mask]
    style_labels = [meta[i]["style_label"] for i in joint_mask]
    le_style = LabelEncoder()
    y_style = le_style.fit_transform(style_labels)

    probe_style = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")
    style_scores = cross_val_score(probe_style, X_joint, y_style, cv=cv, scoring="accuracy")

    probe_style.fit(X_joint, y_style)
    W_style = probe_style.coef_  # [n_styles, 768]

    # --- Subspace overlap metric ---
    genre_dir = W_genre.mean(axis=0, keepdims=True)  # [1, 768]
    style_dir = W_style.mean(axis=0, keepdims=True)  # [1, 768]
    overlap = cosine_similarity(genre_dir, style_dir)[0, 0]

    # Per-class direction overlap matrix
    pairwise_overlap = cosine_similarity(W_genre, W_style)  # [n_genres, n_styles]

    # --- Save results ---
    results = {
        "genre_accuracy_mean": float(genre_scores.mean()),
        "genre_accuracy_std": float(genre_scores.std()),
        "style_accuracy_mean": float(style_scores.mean()),
        "style_accuracy_std": float(style_scores.std()),
        "subspace_overlap_cosine": float(overlap),
        "genre_classes": le_genre.classes_.tolist(),
        "style_classes": le_style.classes_.tolist(),
    }
    with open("outputs/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    np.save("outputs/W_genre.npy", W_genre)
    np.save("outputs/W_style.npy", W_style)
    np.save("outputs/pairwise_overlap.npy", pairwise_overlap)

    # --- Figure: Probe accuracy bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        ["Genre (10-class)", "Style (6-class)"],
        [genre_scores.mean(), style_scores.mean()],
        yerr=[genre_scores.std(), style_scores.std()],
        capsize=5, color=["#2196F3", "#FF9800"],
    )
    ax.set_ylabel("5-Fold CV Accuracy")
    ax.set_title("Linear Probe Accuracy on CoCa Embeddings")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1/10, color="#2196F3", linestyle="--", alpha=0.3, label="Genre chance")
    ax.axhline(y=1/6, color="#FF9800", linestyle="--", alpha=0.3, label="Style chance")
    ax.legend()
    plt.tight_layout()
    plt.savefig("outputs/figures/probe_accuracy.pdf", dpi=300)
    plt.savefig("outputs/figures/probe_accuracy.png", dpi=150)
    print(f"Genre accuracy: {genre_scores.mean():.3f} ± {genre_scores.std():.3f}")
    print(f"Style accuracy: {style_scores.mean():.3f} ± {style_scores.std():.3f}")
    print(f"Subspace overlap (cosine): {overlap:.3f}")

if __name__ == "__main__":
    main()
