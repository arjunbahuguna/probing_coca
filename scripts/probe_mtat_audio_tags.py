"""Run linear probes on MTAT audio embeddings with artist-disjoint folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("outputs/mtat_tasks.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("outputs/mtat_audio_embeddings.npy"))
    parser.add_argument("--metadata", type=Path, default=Path("outputs/mtat_audio_metadata.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mtat_audio_probe_results.json"))
    return parser.parse_args()


def balanced_accuracy_no_warning(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    recalls = []
    for c in range(n_classes):
        class_mask = y_true == c
        support = int(class_mask.sum())
        if support == 0:
            continue
        correct = int((y_pred[class_mask] == c).sum())
        recalls.append(correct / support)
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def evaluate_task(task: dict[str, object], clip_to_embedding: dict[str, np.ndarray]) -> dict[str, object] | None:
    samples = task["samples"]
    kept_samples = [s for s in samples if s["clip_id"] in clip_to_embedding]
    original_to_kept = {
        original_idx: kept_idx
        for kept_idx, original_idx in enumerate(
            i for i, s in enumerate(samples) if s["clip_id"] in clip_to_embedding
        )
    }
    if len(kept_samples) < 4:
        return None

    X = np.stack([clip_to_embedding[s["clip_id"]] for s in kept_samples], axis=0)
    labels = [s["label"] for s in kept_samples]

    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    classes = enc.classes_.tolist()
    if len(classes) < 2:
        return None

    fold_metrics = []
    for fold in task["folds"]:
        train_idx = [original_to_kept[i] for i in fold["train_indices"] if i in original_to_kept]
        test_idx = [original_to_kept[i] for i in fold["test_indices"] if i in original_to_kept]
        if len(train_idx) < 2 or len(test_idx) < 2:
            continue

        y_train = y[train_idx]
        y_test = y[test_idx]
        if len(np.unique(y_train)) < 2:
            continue

        clf = LogisticRegression(max_iter=2000, solver="lbfgs")
        clf.fit(X[train_idx], y_train)
        y_pred = clf.predict(X[test_idx])

        bal_acc = balanced_accuracy_no_warning(y_test, y_pred, n_classes=len(classes))

        fold_metrics.append(
            {
                "fold": int(fold["fold"]),
                "accuracy": float(accuracy_score(y_test, y_pred)),
                "balanced_accuracy": bal_acc,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    if not fold_metrics:
        return None

    accuracies = np.array([m["accuracy"] for m in fold_metrics], dtype=np.float32)
    bal_accuracies = np.array([m["balanced_accuracy"] for m in fold_metrics], dtype=np.float32)

    chance = 1.0 / len(classes)
    class_counts = {c: int(labels.count(c)) for c in classes}

    return {
        "attribute": task["name"],
        "task_type": task["task_type"],
        "classes": classes,
        "num_classes": int(len(classes)),
        "num_samples": int(len(kept_samples)),
        "class_counts": class_counts,
        "cv_splits": int(len(fold_metrics)),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_std": float(accuracies.std()),
        "balanced_accuracy_mean": float(bal_accuracies.mean()),
        "balanced_accuracy_std": float(bal_accuracies.std()),
        "chance_baseline": float(chance),
        "margin_over_chance": float(accuracies.mean() - chance),
        "is_above_chance": bool(accuracies.mean() > chance),
        "fold_metrics": fold_metrics,
    }


def main() -> None:
    args = parse_args()

    with args.tasks.open("r", encoding="utf-8") as f:
        tasks_json = json.load(f)
    with args.metadata.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    X = np.load(args.embeddings)

    clip_to_idx = metadata["clip_id_to_index"]
    clip_to_embedding = {clip_id: X[idx] for clip_id, idx in clip_to_idx.items() if 0 <= idx < len(X)}

    results = []
    skipped = {}
    for task in tasks_json["tasks"]:
        result = evaluate_task(task, clip_to_embedding)
        if result is None:
            skipped[task["name"]] = "insufficient_data_after_filtering"
            continue
        results.append(result)
        print(
            f"{result['attribute']:24s} acc={result['accuracy_mean']:.3f} +/- {result['accuracy_std']:.3f} "
            f"chance={result['chance_baseline']:.3f}"
        )

    results.sort(key=lambda r: r["accuracy_mean"], reverse=True)
    payload = {
        "dataset": "MTAT",
        "condition": "audio",
        "artist_disjoint": True,
        "num_tasks": len(results),
        "results": results,
        "skipped": skipped,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
