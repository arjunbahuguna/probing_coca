# Text vs Audio Separability (Quick Run)

| Attribute | Text Accuracy | Audio Accuracy | Text Chance | Audio Chance | Audio-Text Delta |
|---|---:|---:|---:|---:|---:|
| loudness_proxy | 0.760 +/- 0.049 | 0.930 +/- 0.038 | 0.167 | 0.500 | 0.170 |
| structure_chorus_proxy | 0.993 +/- 0.013 | 0.807 +/- 0.111 | 0.167 | 0.500 | -0.186 |
| timbre_proxy | 0.960 +/- 0.044 | 0.718 +/- 0.100 | 0.200 | 0.250 | -0.242 |
| octave_proxy | 0.740 +/- 0.086 | 0.689 +/- 0.075 | 0.250 | 0.500 | -0.051 |
| tempo_proxy | 0.808 +/- 0.078 | 0.583 +/- 0.064 | 0.200 | 0.333 | -0.225 |
| instrument | 1.000 +/- 0.000 | 0.536 +/- 0.155 | 0.083 | 0.125 | -0.464 |

## Unsupported with MTAT tags

- music_structure_multisection
- chord_choice
- melody
- duration

## Notes

- Audio metrics are artist-disjoint CV from MTAT clips.
- Text metrics come from outputs/additional_probe_results.json.
- Structure is represented only as a chorus proxy when available.
