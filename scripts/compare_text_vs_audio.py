"""Compare text-based and audio-based probe results side by side."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TEXT_MAPPING = {
    "instrument": "instrument",
    "tempo_proxy": "tempo",
    "timbre_proxy": "timbre",
    "loudness_proxy": "loudness",
    "octave_proxy": "octave",
    "structure_chorus_proxy": "music_structure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-results", type=Path, default=Path("outputs/mtat_audio_probe_results.json"))
    parser.add_argument("--text-results", type=Path, default=Path("outputs/additional_probe_results.json"))
    parser.add_argument("--output-json", type=Path, default=Path("outputs/text_vs_audio_comparison.json"))
    parser.add_argument("--output-fig", type=Path, default=Path("outputs/figures/text_vs_audio_comparison.png"))
    parser.add_argument("--output-md", type=Path, default=Path("results_text_vs_audio.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.audio_results.open("r", encoding="utf-8") as f:
        audio_payload = json.load(f)
    with args.text_results.open("r", encoding="utf-8") as f:
        text_results = json.load(f)

    text_by_attr = {r["attribute"]: r for r in text_results}

    rows = []
    for audio_r in audio_payload["results"]:
        audio_attr = audio_r["attribute"]
        text_attr = TEXT_MAPPING.get(audio_attr)
        text_r = text_by_attr.get(text_attr)

        row = {
            "attribute": audio_attr,
            "text_attribute": text_attr,
            "audio_accuracy_mean": audio_r["accuracy_mean"],
            "audio_accuracy_std": audio_r["accuracy_std"],
            "audio_chance": audio_r["chance_baseline"],
            "audio_margin": audio_r["margin_over_chance"],
            "text_accuracy_mean": None,
            "text_accuracy_std": None,
            "text_chance": None,
            "text_margin": None,
            "delta_audio_minus_text": None,
        }
        if text_r is not None:
            row["text_accuracy_mean"] = text_r["accuracy_mean"]
            row["text_accuracy_std"] = text_r["accuracy_std"]
            row["text_chance"] = text_r["chance_baseline"]
            row["text_margin"] = text_r["margin_over_chance"]
            row["delta_audio_minus_text"] = row["audio_accuracy_mean"] - row["text_accuracy_mean"]

        rows.append(row)

    payload = {
        "description": "Quick-run comparison of linear separability for text prompts vs MTAT audio embeddings.",
        "artist_disjoint_audio_eval": True,
        "rows": rows,
        "unsupported_attributes": [
            "music_structure_multisection",
            "chord_choice",
            "melody",
            "duration",
        ],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plottable = [r for r in rows if r["text_accuracy_mean"] is not None]
    if plottable:
        attrs = [r["attribute"] for r in plottable]
        x = np.arange(len(attrs), dtype=np.float32)
        width = 0.36

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.bar(
            x - width / 2,
            [r["text_accuracy_mean"] for r in plottable],
            width=width,
            color="#1565C0",
            alpha=0.9,
            label="Text",
        )
        ax.bar(
            x + width / 2,
            [r["audio_accuracy_mean"] for r in plottable],
            width=width,
            color="#2E7D32",
            alpha=0.9,
            label="Audio (MTAT)",
        )
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel("Cross-Validated Accuracy")
        ax.set_title("Text vs Audio Separability in MusicCoCa Embedding Space")
        ax.set_xticks(x)
        ax.set_xticklabels(attrs, rotation=25, ha="right")
        ax.legend()
        plt.tight_layout()

        args.output_fig.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(args.output_fig, dpi=180)

    lines = [
        "# Text vs Audio Separability (Quick Run)",
        "",
        "| Attribute | Text Accuracy | Audio Accuracy | Text Chance | Audio Chance | Audio-Text Delta |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        text_acc = "n/a" if r["text_accuracy_mean"] is None else f"{r['text_accuracy_mean']:.3f} +/- {r['text_accuracy_std']:.3f}"
        audio_acc = f"{r['audio_accuracy_mean']:.3f} +/- {r['audio_accuracy_std']:.3f}"
        text_chance = "n/a" if r["text_chance"] is None else f"{r['text_chance']:.3f}"
        audio_chance = f"{r['audio_chance']:.3f}"
        delta = "n/a" if r["delta_audio_minus_text"] is None else f"{r['delta_audio_minus_text']:.3f}"
        lines.append(
            f"| {r['attribute']} | {text_acc} | {audio_acc} | {text_chance} | {audio_chance} | {delta} |"
        )

    lines.extend(
        [
            "",
            "## Unsupported with MTAT tags",
            "",
            "- music_structure_multisection",
            "- chord_choice",
            "- melody",
            "- duration",
            "",
            "## Notes",
            "",
            "- Audio metrics are artist-disjoint CV from MTAT clips.",
            "- Text metrics come from outputs/additional_probe_results.json.",
            "- Structure is represented only as a chorus proxy when available.",
        ]
    )

    with args.output_md.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved -> {args.output_json}")
    print(f"Saved -> {args.output_md}")
    if plottable:
        print(f"Saved -> {args.output_fig}")
    else:
        print("Skipped figure; no attributes had matching text baselines.")


if __name__ == "__main__":
    main()
