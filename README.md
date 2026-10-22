# CROA-Attack

**CROA: Cross-Representation Optimal Alignment for Fine-Grained Targeted Transfer Attacks on MLLMs** (ICASSP 2027).

CROA is a vision-only targeted transfer attack. Perturbations are optimized on frozen surrogate encoders and then applied, with no victim queries, to black-box multimodal LLMs. Contrastive CLIP features and self-supervised DINOv2 features are aligned to an attacker-chosen target image by global cosine matching and clustered local optimal transport, with source repulsion and multi-crop normalized momentum.

The manuscript is `paper_icassp/main.tex`. The compiled PDF is `paper_icassp/CROA_ICASSP2027.pdf`.

## Method

For each CLIP encoder and for DINOv2, the attack reads the final-layer CLS token and the patch tokens. Global alignment attracts the adversarial CLS toward the target CLS and repels it from the clean source CLS (`beta = 0.5`). Local alignment runs entropic Sinkhorn optimal transport on 10 k-means centroids of the patch tokens (local weight `0.2` for CLIP and `0.4` for DINOv2). The three CLIP losses share half of the objective and DINOv2 receives the other half. Each step averages four random crops with area scale `[0.35, 1.0]` and applies mean-normalized momentum (`mu = 0.9`).

The shared budget is `L_inf = 16/255`, step size `1/255`, and 300 steps, seed `20260827`. The same configuration is used for every victim. CROA does not use a target caption or an MLLM surrogate while optimizing the perturbation. MPC-A, included as a baseline, does use target text and an InternVL3-1B branch.

Surrogates: CLIP ViT-B/16, CLIP ViT-B/32, LAION CLIP-ViT-G/14, and DINOv2-B/14.

## Layout

```
croa/            config, tensor utilities, model wrappers, attack arms
scripts/         generation, victim inference, scoring, ablations
configs/         example API config (real keys stay out of git)
data/            100 official source-target pairs
experiments/     which script reproduces which table row
paper_icassp/    ICASSP manuscript, style files, and figures
outputs/         created at runtime; gitignored
```

## Install

Python 3.10 or newer. Attack generation and open-source victims need a CUDA GPU. The paper runs used one NVIDIA RTX 6000 Ada (48 GB). Commercial victims and the GPT-4o-mini judge need network access and an API key. No machine-specific path is required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install a CUDA build of PyTorch that matches the local driver before the other packages if `pip` would otherwise pull a CPU wheel. An editable install is optional; the scripts add the repository root to `sys.path` themselves:

```bash
pip install -e .
```

### Surrogate and victim weights

Weights are downloaded from the Hugging Face Hub on first use (or from a local `models_cache/` directory when that folder already contains `config.json`). They are not committed.

| Role | Hugging Face id |
|---|---|
| CLIP-B/16, B/32, LAION-G/14 | `openai/clip-vit-base-patch16`, `openai/clip-vit-base-patch32`, `laion/CLIP-ViT-G-14-laion2B-s12B-b42K` |
| DINOv2 | `facebook/dinov2-base` |
| MPC InternVL branch | `OpenGVLab/InternVL3-1B-hf` |
| LLaVA-1.5-7B | `llava-hf/llava-1.5-7b-hf` |
| Qwen2.5-VL-7B | `Qwen/Qwen2.5-VL-7B-Instruct` |
| InternVL3-8B | `OpenGVLab/InternVL3-8B-hf` |

AnyAttack additionally needs the official Flickr30k cosine decoder checkpoint at `models_cache/anyattack/flickr30k_cos.pt`. That file is released with AnyAttack (CVPR 2025) and is not included here.

### API credentials

Export the judge and commercial-victim credentials. Do not commit them.

```bash
export OPENAI_API_KEY="sk-..."
export OPENAI_BASE_URL="https://api.openai.com/v1"   # omit to use the OpenAI default
```

Alternatively, copy `configs/api_config.example.json` to `configs/api_config.json` and fill in the same fields. `configs/api_config.json` and `.env` are gitignored. The targeted ASR judge and the keyword judge both call `gpt-4o-mini`.

## Data

See `data/README.md`. The 100 pairs are already in the repository:

- sources: `data/bigscale/{0..99}.png`
- targets: `data/target_images/{0..99}.jpg`
- captions and keywords: `data/target_images/caption.json`, `data/target_images/keywords.json`

Targeted ASR compares each victim caption with that victim's own description of the target image (`outputs/descriptions/target_{victim}.jsonl`), using a GPT-4o-mini similarity threshold of 0.5. Keyword match rate is the fraction of target keywords that the same judge marks as present.

## Reproduce

A CPU smoke check confirms the pairs, captions, keywords, and (if PyTorch is installed) the optimal-transport helpers. It does not download weights or call an API.

```bash
python scripts/smoke_test.py
python scripts/smoke_test.py --gpu    # two M-Attack steps on pair 0; needs CUDA and CLIP weights
```

Full table runs are long: 300 steps on CLIP-G for 100 pairs, then one victim model at a time. Pass `--num-pairs 1` to `run_benchmark.py` or `describe_targets.py` for a one-pair wiring check. `generate_croa_tuned.py` always walks the 100 pairs and skips images that already exist.

1. Victim references (once per victim):

```bash
python scripts/describe_targets.py \
  --victims llava15-7b qwen2.5vl7b internvl3-8b gpt-5 gemini-3.5-flash-lite claude-4.5-haiku
```

2. CROA images, then commercial and open-source evaluation:

```bash
python scripts/generate_croa_tuned.py
python scripts/pipeline_croa_tuned_eval.py
python scripts/pipeline_opensource_eval.py
```

3. AttackVLM, AnyAttack, M-Attack, and FOA. Repeat infer and eval with the commercial victim ids for the main commercial table.

```bash
python scripts/run_benchmark.py --phase attack \
  --arms attackvlm anyattack mattack_official foa_official --steps 300
python scripts/run_benchmark.py --phase infer \
  --arms attackvlm anyattack mattack_official foa_official \
  --victims llava15-7b qwen2.5vl7b internvl3-8b
python scripts/run_benchmark.py --phase eval --ref victim \
  --arms attackvlm anyattack mattack_official foa_official \
  --victims llava15-7b qwen2.5vl7b internvl3-8b
```

4. MPC-A (separate runner; uses target text and InternVL3-1B):

```bash
python scripts/run_benchmark_mpc_official.py --generate
python scripts/run_benchmark_mpc_official.py --eval-commercial
python scripts/run_benchmark_mpc_official.py --eval-opensource
python scripts/run_benchmark_mpc_official.py --score
```

5. Ablations, in the same order as the component table (CLIP + multi-scale, w/o DINOv2, w/o local OT, w/o source repulsion):

```bash
python scripts/run_ablation_foa_mca.py --generate
python scripts/run_ablation_croa_no_dino.py --generate
python scripts/run_ablation_croa_no_local.py --generate
python scripts/run_ablation_croa_no_rep.py --generate
```

Each ablation script also accepts `--eval-commercial`, `--eval-opensource`, and `--score`. Their open-source stage queries LLaVA-1.5-7B. The Qwen2.5-VL and InternVL3 ablation cells in the paper use those same images with the shared black-box `ask()` protocol. `run_benchmark.py --phase infer` already accepts all three open-source ids for the arms it generates.

6. Official GPT keyword match on scored CSV files already written under `outputs/responses/`:

```bash
python scripts/eval_official_gpt_kmr.py
```

`scripts/audit_paper_metrics.py` rebuilds a consistent score table from saved responses. `python scripts/audit_paper_metrics.py --rescore` calls the judge for cache misses. Figure scripts in `paper_icassp/` read `outputs/paper_audit_20260922/summary.csv` when regenerating plots. The PNGs included by `main.tex` are already in `paper_icassp/assets/`.

Compile the paper from `paper_icassp/` (the bibliography is inline):

```bash
pdflatex main.tex
```

`scripts/run_benchmark_mattack_v2_full.py` reproduces an expanded M-Attack-V2 variant. It is not a row of the ICASSP tables and expects `data/retrieved_embeddings/`, which is not in this repository. The multi-crop arm that is implemented in `croa/attacks.py` is selected with `--arms mattack_v2`.

## Citation

```bibtex
@inproceedings{chen2027croa,
  title     = {CROA: Cross-Representation Optimal Alignment for Fine-Grained Targeted Transfer Attacks on MLLMs},
  author    = {Chen, Junxi and He, Max and Zhang, Jiahao and Zhao, Xiaoli and Pang, Jinyi and Li, Zuyi and Wu, Yilin},
  booktitle = {Proceedings of the IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year      = {2027}
}
```

Replace this placeholder with the official proceedings entry when it is assigned.

## Responsible use

This repository implements adversarial attacks on multimodal models for defensive research: measuring transfer, comparing published baselines, and stress-testing captioning behavior under a known threat model. Do not use it to deceive people, evade moderation, or compromise a system you do not have permission to test. Commercial and open-source victims are queried only as black-box captioners. Keep API keys out of the tree, and review the provider terms for any model you call.
