"""Generate the prompt taxonomy and save metadata."""
import json
import os
import itertools

GENRES = [
    "deep house", "techno", "jazz", "classical", "heavy metal",
    "reggae", "hip-hop", "bossa nova", "ambient", "drum and bass",
]
STYLES = [
    "energetic", "melancholic", "lo-fi", "aggressive", "dreamy", "minimalist",
]
TEMPLATES = [
    lambda g, s: f"{g}",
    lambda g, s: f"{s} {g}",
    lambda g, s: f"{g} with {s} atmosphere",
]
GENRE_PARAPHRASES = [
    lambda g: f"{g}",
    lambda g: f"{g} music",
    lambda g: f"a {g} track",
    lambda g: f"{g} song",
    lambda g: f"playing {g}",
    lambda g: f"{g} style",
    lambda g: f"{g} beat",
    lambda g: f"{g} sound",
    lambda g: f"the sound of {g}",
    lambda g: f"{g} vibe",
]

def main():
    os.makedirs("outputs", exist_ok=True)
    prompts = []

    # Joint prompts: 10 genres × 6 styles × 3 templates = 180
    for genre, style in itertools.product(GENRES, STYLES):
        for tid, template in enumerate(TEMPLATES):
            prompts.append({
                "text": template(genre, style),
                "genre_label": genre,
                "style_label": style,
                "template_id": tid,
                "set": "joint",
            })

    # Genre-only prompts: 10 genres × 10 paraphrases = 100
    for genre in GENRES:
        for pid, paraphrase in enumerate(GENRE_PARAPHRASES):
            prompts.append({
                "text": paraphrase(genre),
                "genre_label": genre,
                "style_label": None,
                "template_id": pid,
                "set": "genre_only",
            })

    with open("outputs/metadata.json", "w") as f:
        json.dump(prompts, f, indent=2)
    print(f"Generated {len(prompts)} prompts → outputs/metadata.json")

if __name__ == "__main__":
    main()
