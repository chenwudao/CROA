# Experiment entrypoints

Run every command from the repository root after `pip install -r requirements.txt`.
Generated images, captions, and scores go to `outputs/` (gitignored).
The frozen budget is `eps = 16/255`, `steps = 300`, `alpha = 1/255`, seed `20260827`.

`python scripts/smoke_test.py` checks the 100 pairs and, when PyTorch is
installed, the Sinkhorn / k-means helpers. It does not need a GPU or an API key.
`python scripts/smoke_test.py --gpu` runs two steps of M-Attack on pair 0.

| Paper row | Generate | Evaluate |
|---|---|---|
| CROA (ours) | `python scripts/generate_croa_tuned.py` | Commercial: `python scripts/pipeline_croa_tuned_eval.py`. Open-source: `python scripts/pipeline_opensource_eval.py` |
| AttackVLM, AnyAttack, M-Attack, FOA | `python scripts/run_benchmark.py --phase attack --arms attackvlm anyattack mattack_official foa_official` | `--phase infer` then `--phase eval --ref victim` on the same `--arms` and `--victims` |
| MPC-A | `python scripts/run_benchmark_mpc_official.py --generate` | `--eval-commercial`, `--eval-opensource`, `--score` |
| CLIP + multi-scale | `python scripts/run_ablation_foa_mca.py --generate` | `--eval-commercial`, `--eval-opensource`, `--score` |
| CROA w/o DINOv2 | `python scripts/run_ablation_croa_no_dino.py --generate` | same three flags |
| CROA w/o local OT | `python scripts/run_ablation_croa_no_local.py --generate` | same three flags |
| CROA w/o repulsion | `python scripts/run_ablation_croa_no_rep.py --generate` | same three flags |

Before scoring, build victim references with:

```bash
python scripts/describe_targets.py --victims llava15-7b qwen2.5vl7b internvl3-8b gpt-5 gemini-3.5-flash-lite claude-4.5-haiku
```

Official GPT keyword-match rescoring of existing `outputs/responses/scored_*.csv` files:

```bash
python scripts/eval_official_gpt_kmr.py
```

Open-source victim ids: `llava15-7b`, `qwen2.5vl7b`, `internvl3-8b`.
Commercial victim ids: `gpt-5`, `gemini-3.5-flash-lite`, `claude-4.5-haiku`.

The ablation runners and `run_benchmark_mpc_official.py` query LLaVA-1.5-7B
in their open-source stage. Qwen and InternVL3 numbers in the paper use the
same saved images and the same `make_victim(...).ask(...)` protocol.
`run_benchmark.py --phase infer` accepts all three open-source ids.

`scripts/run_benchmark_mattack_v2_full.py` is the expanded M-Attack-V2
reproduction. It is not a row of the ICASSP main tables and needs
`data/retrieved_embeddings/`, which is not shipped here. The restricted
multi-crop arm inside `croa/attacks.py` is `mattack_v2`.
