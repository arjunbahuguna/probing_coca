"""Prepare MTAT quick-run tasks with artist-disjoint CV folds."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.model_selection import GroupKFold


UNSUPPORTED_ATTRIBUTES = [
    "music_structure_multisection",
    "chord_choice",
    "melody",
    "duration",
]


MULTICLASS_FAMILIES = {
    "instrument": {
        "task_type": "multiclass",
        "class_to_tags": {
            "guitar": ["guitar", "acoustic guitar", "electric guitar", "classical guitar"],
            "piano": ["piano", "piano solo"],
            "drums": ["drum", "drums", "percussion"],
            "violin": ["violin", "violins"],
            "flute": ["flute", "flutes"],
            "sax": ["sax"],
            "trumpet": ["trumpet", "horn", "horns"],
            "synthesizer": ["synth", "synthesizer", "electro", "electronica"],
        },
    },
    "tempo_proxy": {
        "task_type": "multiclass",
        "class_to_tags": {
            "slow": ["slow"],
            "fast": ["fast", "quick", "fast beat"],
            "upbeat": ["upbeat"],
        },
    },
    "timbre_proxy": {
        "task_type": "multiclass",
        "class_to_tags": {
            "airy": ["airy"],
            "dark": ["dark"],
            "soft": ["soft", "mellow"],
            "hard": ["hard", "heavy"],
        },
    },
}


BINARY_TASKS = {
    "loudness_proxy": {
        "task_type": "binary",
        "positive_tags": ["loud", "quiet", "soft", "heavy"],
        "strategy": "loud_vs_quiet",
    },
    "octave_proxy": {
        "task_type": "binary",
        "positive_tags": ["low"],
        "positive_label": "low_register",
        "negative_label": "not_low_register",
    },
    "structure_chorus_proxy": {
        "task_type": "binary",
        "positive_tags": ["chorus"],
        "positive_label": "chorus_present",
        "negative_label": "chorus_absent",
    },
}


@dataclass
class ClipRow:
    clip_id: str
    mp3_path: str
    artist: str
    artist_id: str
    tags: set[str]
    audio_exists: bool


def _clean(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        value = value[1:-1]
    return value.strip()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows: list[dict[str, str]] = []
        for row in reader:
            cleaned = {_clean(k): _clean(v or "") for k, v in row.items() if k is not None}
            rows.append(cleaned)
    return rows


def artist_id_from_mp3_path(mp3_path: str) -> str:
    stem = Path(mp3_path).stem
    return stem.split("-")[0]


def build_clip_rows(
    annotations: list[dict[str, str]],
    clip_info: list[dict[str, str]],
    mtat_root: Path,
    require_audio: bool,
) -> list[ClipRow]:
    info_by_clip_id = {r["clip_id"]: r for r in clip_info}
    rows: list[ClipRow] = []

    for row in annotations:
        clip_id = row.get("clip_id", "")
        if not clip_id:
            continue
        mp3_path = row.get("mp3_path", "")
        if not mp3_path:
            continue

        full_path = mtat_root / mp3_path
        audio_exists = full_path.exists()
        if require_audio and not audio_exists:
            continue

        artist = info_by_clip_id.get(clip_id, {}).get("artist", "")
        if not artist:
            artist = artist_id_from_mp3_path(mp3_path)

        tags = {k for k, v in row.items() if k not in {"clip_id", "mp3_path"} and v == "1"}
        rows.append(
            ClipRow(
                clip_id=clip_id,
                mp3_path=mp3_path,
                artist=artist,
                artist_id=artist.lower().strip() or artist_id_from_mp3_path(mp3_path),
                tags=tags,
                audio_exists=audio_exists,
            )
        )
    return rows


def sample_balanced(ids_by_class: dict[str, list[int]], cap_per_class: int, rng: random.Random) -> dict[str, list[int]]:
    min_count = min(len(v) for v in ids_by_class.values())
    target = min(min_count, cap_per_class)
    sampled: dict[str, list[int]] = {}
    for label, ids in ids_by_class.items():
        shuffled = ids[:]
        rng.shuffle(shuffled)
        sampled[label] = shuffled[:target]
    return sampled


def build_group_folds(labels: Sequence[str], artists: Sequence[str], n_splits: int) -> list[dict[str, object]]:
    unique_artists = len(set(artists))
    splits = min(n_splits, unique_artists)
    if splits < 2:
        return []

    X = np.zeros((len(labels), 1), dtype=np.float32)
    y = np.array(labels)
    groups = np.array(artists)

    gkf = GroupKFold(n_splits=splits)
    folds: list[dict[str, object]] = []
    for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=groups)):
        train_artists = set(groups[train_idx].tolist())
        test_artists = set(groups[test_idx].tolist())
        if train_artists & test_artists:
            raise AssertionError("Artist overlap detected in supposedly disjoint fold")
        folds.append(
            {
                "fold": fold_idx,
                "train_indices": train_idx.tolist(),
                "test_indices": test_idx.tolist(),
                "train_artist_count": len(train_artists),
                "test_artist_count": len(test_artists),
            }
        )
    return folds


def build_multiclass_task(
    task_name: str,
    class_to_tags: dict[str, list[str]],
    rows: list[ClipRow],
    cap_per_class: int,
    min_per_class: int,
    n_splits: int,
    rng: random.Random,
) -> tuple[dict[str, object] | None, str | None]:
    ids_by_class: dict[str, list[int]] = {label: [] for label in class_to_tags}
    for idx, row in enumerate(rows):
        matched = []
        for label, tags in class_to_tags.items():
            if any(t in row.tags for t in tags):
                matched.append(label)
        if len(matched) == 1:
            ids_by_class[matched[0]].append(idx)

    if min(len(v) for v in ids_by_class.values()) < min_per_class:
        return None, f"insufficient_support<{min_per_class}"

    sampled = sample_balanced(ids_by_class, cap_per_class, rng)
    samples = []
    for label, ids in sampled.items():
        for i in ids:
            row = rows[i]
            samples.append(
                {
                    "clip_id": row.clip_id,
                    "mp3_path": row.mp3_path,
                    "artist": row.artist,
                    "artist_id": row.artist_id,
                    "label": label,
                }
            )

    rng.shuffle(samples)
    labels = [s["label"] for s in samples]
    artists = [s["artist_id"] for s in samples]
    folds = build_group_folds(labels, artists, n_splits=n_splits)
    if not folds:
        return None, "insufficient_unique_artists"

    return (
        {
            "name": task_name,
            "task_type": "multiclass",
            "classes": list(class_to_tags.keys()),
            "samples": samples,
            "folds": folds,
            "num_samples": len(samples),
        },
        None,
    )


def build_loud_vs_quiet_task(
    rows: list[ClipRow],
    cap_per_class: int,
    min_per_class: int,
    n_splits: int,
    rng: random.Random,
) -> tuple[dict[str, object] | None, str | None]:
    loud_ids = []
    quiet_ids = []
    for idx, row in enumerate(rows):
        has_loud = "loud" in row.tags
        has_quiet = "quiet" in row.tags or "soft" in row.tags
        if has_loud and not has_quiet:
            loud_ids.append(idx)
        elif has_quiet and not has_loud:
            quiet_ids.append(idx)

    if min(len(loud_ids), len(quiet_ids)) < min_per_class:
        return None, f"insufficient_support<{min_per_class}"

    sampled = sample_balanced({"loud": loud_ids, "quiet": quiet_ids}, cap_per_class, rng)
    samples = []
    for label, ids in sampled.items():
        for i in ids:
            row = rows[i]
            samples.append(
                {
                    "clip_id": row.clip_id,
                    "mp3_path": row.mp3_path,
                    "artist": row.artist,
                    "artist_id": row.artist_id,
                    "label": label,
                }
            )

    rng.shuffle(samples)
    labels = [s["label"] for s in samples]
    artists = [s["artist_id"] for s in samples]
    folds = build_group_folds(labels, artists, n_splits=n_splits)
    if not folds:
        return None, "insufficient_unique_artists"

    return (
        {
            "name": "loudness_proxy",
            "task_type": "binary",
            "classes": ["loud", "quiet"],
            "samples": samples,
            "folds": folds,
            "num_samples": len(samples),
        },
        None,
    )


def build_binary_task(
    task_name: str,
    positive_tags: Iterable[str],
    positive_label: str,
    negative_label: str,
    rows: list[ClipRow],
    cap_per_class: int,
    min_per_class: int,
    n_splits: int,
    rng: random.Random,
) -> tuple[dict[str, object] | None, str | None]:
    positive_ids = []
    negative_ids = []
    positive_tag_set = set(positive_tags)
    for idx, row in enumerate(rows):
        if row.tags & positive_tag_set:
            positive_ids.append(idx)
        else:
            negative_ids.append(idx)

    if min(len(positive_ids), len(negative_ids)) < min_per_class:
        return None, f"insufficient_support<{min_per_class}"

    sampled = sample_balanced({positive_label: positive_ids, negative_label: negative_ids}, cap_per_class, rng)
    samples = []
    for label, ids in sampled.items():
        for i in ids:
            row = rows[i]
            samples.append(
                {
                    "clip_id": row.clip_id,
                    "mp3_path": row.mp3_path,
                    "artist": row.artist,
                    "artist_id": row.artist_id,
                    "label": label,
                }
            )

    rng.shuffle(samples)
    labels = [s["label"] for s in samples]
    artists = [s["artist_id"] for s in samples]
    folds = build_group_folds(labels, artists, n_splits=n_splits)
    if not folds:
        return None, "insufficient_unique_artists"

    return (
        {
            "name": task_name,
            "task_type": "binary",
            "classes": [positive_label, negative_label],
            "samples": samples,
            "folds": folds,
            "num_samples": len(samples),
        },
        None,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mtat-root", type=Path, default=Path("mtat"))
    parser.add_argument("--annotations", type=Path, default=Path("mtat/annotations_final.csv"))
    parser.add_argument("--clip-info", type=Path, default=Path("mtat/clip_info_final.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/mtat_tasks.json"))
    parser.add_argument("--max-clips-per-class", type=int, default=120)
    parser.add_argument("--min-clips-per-class", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-missing-audio", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    annotations = read_tsv(args.annotations)
    clip_info = read_tsv(args.clip_info)
    rows = build_clip_rows(
        annotations=annotations,
        clip_info=clip_info,
        mtat_root=args.mtat_root,
        require_audio=not args.allow_missing_audio,
    )

    task_entries: list[dict[str, object]] = []
    skipped: dict[str, str] = {}

    for task_name, spec in MULTICLASS_FAMILIES.items():
        task, reason = build_multiclass_task(
            task_name=task_name,
            class_to_tags=spec["class_to_tags"],
            rows=rows,
            cap_per_class=args.max_clips_per_class,
            min_per_class=args.min_clips_per_class,
            n_splits=args.n_splits,
            rng=rng,
        )
        if task is None:
            skipped[task_name] = reason or "unknown"
        else:
            task_entries.append(task)

    loudness_task, loudness_reason = build_loud_vs_quiet_task(
        rows=rows,
        cap_per_class=args.max_clips_per_class,
        min_per_class=args.min_clips_per_class,
        n_splits=args.n_splits,
        rng=rng,
    )
    if loudness_task is None:
        skipped["loudness_proxy"] = loudness_reason or "unknown"
    else:
        task_entries.append(loudness_task)

    for task_name, spec in BINARY_TASKS.items():
        if task_name == "loudness_proxy":
            continue
        task, reason = build_binary_task(
            task_name=task_name,
            positive_tags=spec["positive_tags"],
            positive_label=spec["positive_label"],
            negative_label=spec["negative_label"],
            rows=rows,
            cap_per_class=args.max_clips_per_class,
            min_per_class=args.min_clips_per_class,
            n_splits=args.n_splits,
            rng=rng,
        )
        if task is None:
            skipped[task_name] = reason or "unknown"
        else:
            task_entries.append(task)

    args.output.parent.mkdir(parents=True, exist_ok=True)

    output = {
        "dataset": "MTAT",
        "quick_run": True,
        "seed": args.seed,
        "constraints": {
            "artist_disjoint": True,
            "max_clips_per_class": args.max_clips_per_class,
            "min_clips_per_class": args.min_clips_per_class,
            "n_splits": args.n_splits,
            "require_audio": not args.allow_missing_audio,
        },
        "unsupported_attributes": UNSUPPORTED_ATTRIBUTES,
        "task_count": len(task_entries),
        "tasks": task_entries,
        "skipped_tasks": skipped,
        "source_files": {
            "annotations": os.fspath(args.annotations),
            "clip_info": os.fspath(args.clip_info),
        },
        "dataset_stats": {
            "rows_considered": len(rows),
            "unique_artists": len({r.artist_id for r in rows}),
            "audio_available_rows": int(sum(1 for r in rows if r.audio_exists)),
        },
    }

    with args.output.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved task file: {args.output}")
    print(f"Tasks kept: {len(task_entries)}")
    if skipped:
        print(f"Skipped tasks: {len(skipped)}")
        for name, reason in skipped.items():
            print(f"  - {name}: {reason}")


if __name__ == "__main__":
    main()
