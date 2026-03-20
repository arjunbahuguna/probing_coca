"""Extract CoCa embeddings for all prompts."""
import json
import numpy as np
import sys
import os

# Add magenta-realtime to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "magenta-realtime"))

from magenta_rt import musiccoca  # actual import path

def main():
    # Load prompts
    with open("outputs/metadata.json") as f:
        prompts = json.load(f)

    # Initialize MusicCoCa (auto-downloads assets on first use)
    model = musiccoca.MusicCoCa()  # = MusicCoCaV212F(), lazy=True

    # Extract embeddings in batches
    # embed_batch_text expects List[str], returns np.ndarray[B, 768]
    texts = [p["text"] for p in prompts]
    BATCH_SIZE = 32
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        # Uses MusicCoCaBase.embed_batch_text → _embed_batch_text
        # which calls self._encoder.signatures['embed_text']
        embs = model.embed_batch_text(batch)  # np.ndarray [B, 768]
        all_embeddings.append(embs)
        print(f"  Batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1} done")

    embeddings = np.concatenate(all_embeddings, axis=0)  # [N, 768]
    np.save("outputs/embeddings.npy", embeddings)
    print(f"Saved embeddings: {embeddings.shape} → outputs/embeddings.npy")

if __name__ == "__main__":
    main()
