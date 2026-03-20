"""Extract MusicCoCa audio embeddings for MTAT quick-run tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

# Add magenta-realtime to path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "magenta-realtime"))
from magenta_rt import audio, musiccoca


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, default=Path("outputs/mtat_tasks.json"))
    parser.add_argument("--mtat-root", type=Path, default=Path("mtat"))
    parser.add_argument("--embeddings-out", type=Path, default=Path("outputs/mtat_audio_embeddings.npy"))
    parser.add_argument("--metadata-out", type=Path, default=Path("outputs/mtat_audio_metadata.json"))
    parser.add_argument("--max-clips", type=int, default=None)
    return parser.parse_args()


def load_unique_samples(tasks_json: dict[str, object]) -> list[dict[str, str]]:
    sample_by_id: dict[str, dict[str, str]] = {}
    for task in tasks_json["tasks"]:
        for sample in task["samples"]:
            clip_id = sample["clip_id"]
            if clip_id not in sample_by_id:
                sample_by_id[clip_id] = {
                    "clip_id": sample["clip_id"],
                    "mp3_path": sample["mp3_path"],
                    "artist": sample["artist"],
                    "artist_id": sample["artist_id"],
                }
    return list(sample_by_id.values())


def main() -> None:
    args = parse_args()

    with args.tasks.open("r", encoding="utf-8") as f:
        tasks_json = json.load(f)

    samples = load_unique_samples(tasks_json)
    samples.sort(key=lambda x: x["clip_id"])

    if args.max_clips is not None:
        samples = samples[: args.max_clips]

    missing = []
    for sample in samples:
        full_path = args.mtat_root / sample["mp3_path"]
        if not full_path.exists():
            missing.append(sample["mp3_path"])
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            "Preflight failed. Missing audio clips referenced by task file. "
            f"Missing count={len(missing)}.\nFirst missing entries:\n{preview}"
        )

    model = musiccoca.MusicCoCa()
    vectors = []
    clip_metadata = []

    for i, sample in enumerate(samples):
        audio_path = args.mtat_root / sample["mp3_path"]
        waveform = audio.Waveform.from_file(os.fspath(audio_path))
        emb = model.embed(waveform)
        vectors.append(np.asarray(emb, dtype=np.float32))
        clip_metadata.append(
            {
                "index": i,
                "clip_id": sample["clip_id"],
                "mp3_path": sample["mp3_path"],
                "artist": sample["artist"],
                "artist_id": sample["artist_id"],
            }
        )
        if (i + 1) % 50 == 0 or i + 1 == len(samples):
            print(f"Embedded {i + 1}/{len(samples)} clips")

    X = np.stack(vectors, axis=0)

    args.embeddings_out.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.embeddings_out, X)

    metadata = {
        "source_tasks": os.fspath(args.tasks),
        "mtat_root": os.fspath(args.mtat_root),
        "num_clips": int(X.shape[0]),
        "embedding_dim": int(X.shape[1]),
        "clips": clip_metadata,
        "clip_id_to_index": {m["clip_id"]: m["index"] for m in clip_metadata},
    }
    with args.metadata_out.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved embeddings: {X.shape} -> {args.embeddings_out}")
    print(f"Saved metadata -> {args.metadata_out}")


if __name__ == "__main__":
    main()
