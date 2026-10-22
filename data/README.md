# Benchmark data

The ICASSP experiments use the 100 source–target pairs shared by
[M-Attack](https://github.com/VILA-Lab/M-Attack) and
[FOA-Attack](https://github.com/jiaxiaojunQAQ/FOA-Attack). The paper describes
these pairs as images from the NIPS 2017 Adversarial Attacks and Defenses
competition, with the fine-grained target keywords released by those projects.

This directory already contains that fixed split:

| Path | Role |
|---|---|
| `bigscale/{0..99}.png` | Clean source images |
| `target_images/{0..99}.jpg` | Attacker-chosen target images |
| `target_images/caption.json` | Human reference captions (five strings per target, where provided) |
| `target_images/keywords.json` | Target keywords used for keyword-match rate |

`croa.config.load_pairs()` pairs `bigscale/i.png` with `target_images/i.jpg`.
`caption.json` has no entry for `51.jpg`; `load_captions()` inserts the same
fallback sentences the benchmark has been using. Do not renumber the files.

No adversarial images are stored here. Attack scripts write those under
`outputs/`, which is gitignored.

The expanded M-Attack-V2 runner (`scripts/run_benchmark_mattack_v2_full.py`)
also looks for auxiliary targets in `data/retrieved_embeddings/{id}/`. Those
files are not part of this split and are not required for the tables in
`paper_icassp/main.tex`.
