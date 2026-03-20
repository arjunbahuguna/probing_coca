"""Probe linear separability of additional music attributes in CoCa text embedding space."""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder

# Add magenta-realtime to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "magenta-realtime"))
from magenta_rt import musiccoca


ATTRIBUTE_CLASSES = {
    "tempo": [
        "very slow tempo",
        "slow tempo",
        "medium tempo",
        "fast tempo",
        "very fast tempo",
    ],
    "timbre": [
        "warm timbre",
        "bright timbre",
        "dark timbre",
        "harsh timbre",
        "airy timbre",
    ],
    "music_structure": [
        "intro section",
        "verse section",
        "chorus section",
        "bridge section",
        "outro section",
        "drop section",
    ],
    "chord_choice": [
        "major chords",
        "minor chords",
        "diminished chords",
        "dominant seventh chords",
        "suspended chords",
        "quartal harmony",
    ],
    "melody": [
        "stepwise melody",
        "arpeggiated melody",
        "lyrical melody",
        "repetitive melody",
        "chromatic melody",
    ],
    "octave": [
        "low octave register",
        "mid octave register",
        "high octave register",
        "very high octave register",
    ],
    "duration": [
        "very short duration",
        "short duration",
        "medium duration",
        "long duration",
        "very long duration",
    ],
    "instrument": [
        "piano",
        "electric guitar",
        "acoustic guitar",
        "violin",
        "cello",
        "trumpet",
        "saxophone",
        "flute",
        "drum kit",
        "synthesizer",
        "bass guitar",
        "harp",
    ],
    "loudness": [
        "very quiet volume",
        "quiet volume",
        "moderate volume",
        "loud volume",
        "very loud volume",
        "clipped distortion loudness",
    ],
}

PROMPT_TEMPLATES = [
    "{label} music",
    "a track with {label}",
    "a composition featuring {label}",
    "music characterized by {label}",
    "an arrangement emphasizing {label}",
]

CONTEXTS = [
    "for a soundtrack",
    "for a studio recording",
    "for a live performance",
    "for focused listening",
    "for a dance floor",
]


def build_prompts_for_attribute(attribute_name, labels):
    prompts = []
    for label in labels:
        for template in PROMPT_TEMPLATES:
            for context in CONTEXTS:
                prompts.append(
                    {
                        "text": f"{template.format(label=label)} {context}",
                        "attribute": attribute_name,
                        "label": label,
                    }
                )
    return prompts


def evaluate_attribute(model, attribute_name, labels):
    prompts = build_prompts_for_attribute(attribute_name, labels)
    texts = [p["text"] for p in prompts]
    y_labels = [p["label"] for p in prompts]

    all_embeddings = []
    batch_size = 64
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        emb = model.embed_batch_text(batch)
        all_embeddings.append(emb)
    X = np.concatenate(all_embeddings, axis=0)

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    _, counts = np.unique(y, return_counts=True)
    min_count = int(counts.min())
    n_splits = min(5, min_count)
    if n_splits < 2:
        raise ValueError(f"Not enough samples per class for CV in {attribute_name}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=3000, C=1.0, solver="lbfgs")
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")

    chance = 1.0 / len(labels)
    return {
        "attribute": attribute_name,
        "num_classes": len(labels),
        "num_prompts": len(prompts),
        "cv_splits": n_splits,
        "accuracy_mean": float(scores.mean()),
        "accuracy_std": float(scores.std()),
        "chance_baseline": float(chance),
        "margin_over_chance": float(scores.mean() - chance),
        "is_above_chance": bool(scores.mean() > chance),
        "classes": list(labels),
    }, prompts


def main():
    out_dir = Path("outputs")
    figures_dir = out_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    model = musiccoca.MusicCoCa()

    all_results = []
    all_prompts = []

    for attribute_name, labels in ATTRIBUTE_CLASSES.items():
        result, prompts = evaluate_attribute(model, attribute_name, labels)
        all_results.append(result)
        all_prompts.extend(prompts)
        print(
            f"{attribute_name:16s} acc={result['accuracy_mean']:.3f} +/- {result['accuracy_std']:.3f} "
            f"chance={result['chance_baseline']:.3f}"
        )

    all_results = sorted(all_results, key=lambda x: x["accuracy_mean"], reverse=True)

    with open(out_dir / "additional_attribute_prompts.json", "w") as f:
        json.dump(all_prompts, f, indent=2)

    with open(out_dir / "additional_probe_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    names = [r["attribute"] for r in all_results]
    means = [r["accuracy_mean"] for r in all_results]
    stds = [r["accuracy_std"] for r in all_results]
    chances = [r["chance_baseline"] for r in all_results]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, means, yerr=stds, capsize=4, color="#1976D2", alpha=0.9)
    ax.scatter(x, chances, color="#E64A19", marker="_", s=300, label="Chance baseline")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Cross-Validated Accuracy")
    ax.set_title("Linear Separability of Additional Attributes in CoCa Embeddings")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "additional_probe_accuracy.png", dpi=180)
    plt.savefig(figures_dir / "additional_probe_accuracy.pdf", dpi=300)

    md_lines = [
        "# Additional Attribute Probe Results",
        "",
        "| Attribute | Accuracy (mean +/- std) | Chance | Margin | Above Chance | Classes |",
        "|---|---:|---:|---:|:---:|---:|",
    ]

    for r in all_results:
        md_lines.append(
            f"| {r['attribute']} | {r['accuracy_mean']:.3f} +/- {r['accuracy_std']:.3f} | "
            f"{r['chance_baseline']:.3f} | {r['margin_over_chance']:.3f} | "
            f"{'yes' if r['is_above_chance'] else 'no'} | {r['num_classes']} |"
        )

    md_lines.extend(
        [
            "",
            "## Figure",
            "",
            "- [Additional Probe Accuracy](outputs/figures/additional_probe_accuracy.png)",
            "",
            "## Notes",
            "",
            "- These are text-only prompts, so results reflect separability in CoCa text-conditioning space.",
            "- Strong scores can partly reflect explicit lexical cues in prompts.",
        ]
    )

    with open("results_additional_attributes.md", "w") as f:
        f.write("\n".join(md_lines) + "\n")

    print("Saved outputs/additional_probe_results.json")
    print("Saved outputs/additional_attribute_prompts.json")
    print("Saved outputs/figures/additional_probe_accuracy.(png|pdf)")
    print("Saved results_additional_attributes.md")


if __name__ == "__main__":
    main()
