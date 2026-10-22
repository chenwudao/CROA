"""Benchmark pipeline: MPCAttack adaptation (CVPR 2026; not an official faithful port).

Runs a locally adapted MPCAttack on the 100-pair NeurIPS-2017 benchmark (data/bigscale + data/target_images)
with human target text guidance from caption.json (0.6 img + 0.4 txt fusion).
Unlike the original, source text is not generated/fused. The legacy arm ID
mpc_official is retained only to preserve existing result provenance.
Surrogate ensemble: 5 branches (CLIP-B16, CLIP-B32, CLIP-Laion, DINOv2-base, InternVL3-1B).
Budget: 300 steps, eps = 16/255, MI-FGSM schedule (mu = 0.9).

Victims:
  - gpt-5
  - gemini-3.5-flash-lite
  - claude-4.5-haiku
  - llava15-7b (open-source anchor)

Metrics:
  - Target-ASR (%) [GPTScore >= 0.5 via gpt-4o-mini]
  - Official GPT-KMR (%) [gpt-4o-mini semantic keyword match]
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from croa import config as C
from croa.attacks import MPCOfficial
from scripts.eval_official_gpt_kmr import GPTKeywordMatcher
from croa.models import FeatureBank, InternVLSurrogate, make_victim
from scripts.run_benchmark import GPTScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARM = "mpc_official"
COMMERCIAL_VICTIMS = ["gpt-5", "gemini-3.5-flash-lite", "claude-4.5-haiku"]
OPENSOURCE_VICTIMS = ["llava15-7b"]
ALL_VICTIMS = COMMERCIAL_VICTIMS + OPENSOURCE_VICTIMS

ADV_DIR = ROOT / "outputs" / "adv" / ARM
ADV_DIR.mkdir(parents=True, exist_ok=True)
REC_FILE = ROOT / "outputs" / "records" / f"attack_records_{ARM}.jsonl"
REC_FILE.parent.mkdir(parents=True, exist_ok=True)
RESP_DIR = ROOT / "outputs" / "responses"
RESP_DIR.mkdir(parents=True, exist_ok=True)
DESC_DIR = ROOT / "outputs" / "descriptions"


def generate_adv_images():
    print("\n" + "=" * 60)
    print(f"STAGE 1: Generating 100 Adversarial Images for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)
    captions = C.load_captions()

    bank = FeatureBank(device=DEVICE)
    ivl = InternVLSurrogate(device=DEVICE)
    attacker = MPCOfficial(bank=bank, ivl=ivl, device=DEVICE)

    completed = set()
    if REC_FILE.exists():
        for line in REC_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    completed.add(int(r["sample_id"]))
                except Exception:
                    pass

    t0_all = time.time()
    for i, pair in enumerate(pairs):
        sid = int(pair["sample_id"])
        out_png = ADV_DIR / f"{sid}.png"

        if sid in completed and out_png.exists():
            print(f"[{i+1}/100] Pair {sid} already generated, skipping.")
            continue

        src_pil = Image.open(pair["source"]).convert("RGB")
        tgt_pil = Image.open(pair["target"]).convert("RGB")

        # Get target caption y_star
        target_caps = captions.get(str(sid), [""])
        y_star = target_caps[0] if target_caps else ""

        t0 = time.time()
        res = attacker.attack(
            source_pil=src_pil,
            tgt_pil=tgt_pil,
            steps=300,
            seed=20260827,
            y_star=y_star,
            mpc_text_fusion=True
        )
        elapsed = time.time() - t0

        adv_arr = (res["adv"].cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(adv_arr).save(out_png)

        rec = {
            "arm": ARM,
            "sample_id": sid,
            "final_s_pos": res.get("final_s_pos"),
            "final_s_neg": res.get("final_s_neg"),
            "linf": res["linf"],
            "elapsed_sec": elapsed,
            "timestamp": time.time(),
            "config": {
                "branches": ["clip_B16", "clip_B32", "clip_Laion", "internvl", "dino"],
                "text_fusion": True,
                "lambda_txt": 0.6,
                "tau": 0.2,
                "omega": 2.0,
                "steps": 300,
                "eps": 16.0 / 255.0
            }
        }
        with REC_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[{i+1}/100] Pair {sid:02d} | s_pos={res.get('final_s_pos', 0):.4f} | linf={res['linf']:.4f} | time={elapsed:.1f}s", flush=True)

    print(f"\n[Generation Complete] 100 images saved in {ADV_DIR} (total time: {time.time()-t0_all:.1f}s)")
    del bank
    del ivl
    del attacker
    torch.cuda.empty_cache()


def run_commercial_inference():
    print("\n" + "=" * 60)
    print(f"STAGE 2: Commercial Frontier Inference for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)

    for v_name in COMMERCIAL_VICTIMS:
        resp_file = RESP_DIR / f"responses_{v_name}.jsonl"
        done = set()
        if resp_file.exists():
            for line in resp_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("arm") == ARM:
                            done.add(str(r["sample_id"]))
                    except Exception:
                        pass

        missing = [p for p in pairs if str(p["sample_id"]) not in done]
        print(f"\n[Victim: {v_name}] {len(done)}/100 done, {len(missing)} remaining.")
        sys.stdout.flush()

        if not missing:
            continue

        victim = make_victim(v_name)
        with resp_file.open("a", encoding="utf-8") as fp:
            for i, p in enumerate(missing):
                sid = str(p["sample_id"])
                adv_png = ADV_DIR / f"{sid}.png"
                if not adv_png.exists():
                    print(f"  [ERROR] Missing {adv_png}!")
                    continue

                adv_pil = Image.open(adv_png).convert("RGB")
                cap = "none"
                for attempt in range(5):
                    try:
                        cap = victim.ask(adv_pil, C.CAPTION_PROMPT, max_new_tokens=300) or "none"
                        break
                    except Exception as e:
                        print(f"  [{v_name}|{ARM}] retry {attempt+1} for pair {sid}: {e}", flush=True)
                        time.sleep(3 + 3 * attempt)

                entry = {
                    "arm": ARM,
                    "victim": v_name,
                    "sample_id": sid,
                    "caption": cap,
                    "model_version": v_name,
                    "timestamp": time.time()
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()
                print(f"  [{v_name}|{ARM}] Pair {sid}: {cap[:75]}...", flush=True)


def run_opensource_inference():
    print("\n" + "=" * 60)
    print(f"STAGE 3: Open-Source Inference on LLaVA-1.5-7B for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)
    for v_name in OPENSOURCE_VICTIMS:
        resp_file = RESP_DIR / f"responses_{v_name}.jsonl"
        done = set()
        if resp_file.exists():
            for line in resp_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("arm") == ARM:
                            done.add(str(r["sample_id"]))
                    except Exception:
                        pass

        missing = [p for p in pairs if str(p["sample_id"]) not in done]
        print(f"\n[Victim: {v_name}] {len(done)}/100 done, {len(missing)} remaining.")
        sys.stdout.flush()

        if not missing:
            continue

        victim = make_victim(v_name)
        with resp_file.open("a", encoding="utf-8") as fp:
            for p in missing:
                sid = str(p["sample_id"])
                adv_png = ADV_DIR / f"{sid}.png"
                adv_pil = Image.open(adv_png).convert("RGB")
                cap = victim.ask(adv_pil, C.CAPTION_PROMPT, max_new_tokens=300) or "none"
                entry = {
                    "arm": ARM,
                    "victim": v_name,
                    "sample_id": sid,
                    "caption": cap,
                    "model_version": v_name,
                    "timestamp": time.time()
                }
                fp.write(json.dumps(entry) + "\n")
                fp.flush()
                print(f"  [{v_name}|{ARM}] Pair {sid}: {cap[:75]}...", flush=True)

        del victim
        torch.cuda.empty_cache()


def score_dual_metrics():
    print("\n" + "=" * 60)
    print(f"STAGE 4: Dual-Metric Scoring (Target-ASR + Official GPT-KMR) for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    scorer = GPTScorer()
    matcher = GPTKeywordMatcher()
    keywords = C.load_keywords()
    thr = C.GPTSCORE_THRESHOLD

    summary_rows = []

    for v_name in ALL_VICTIMS:
        resp_file = RESP_DIR / f"responses_{v_name}.jsonl"
        target_file = DESC_DIR / f"target_{v_name}.jsonl"
        csv_p = RESP_DIR / f"scored_{v_name}.csv"

        tgt_refs = {}
        if target_file.exists():
            for line in target_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        tgt_refs[str(r["sample_id"])] = r["caption"]
                    except Exception:
                        pass

        arm_resps = []
        if resp_file.exists():
            for line in resp_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        if r.get("arm") == ARM:
                            arm_resps.append(r)
                    except Exception:
                        pass

        # Deduplicate to latest
        latest_resps = {}
        for r in arm_resps:
            latest_resps[str(r["sample_id"])] = r
        arm_resps = [latest_resps[k] for k in sorted(latest_resps.keys(), key=lambda x: int(x))]

        print(f"\n[Scoring {v_name} | {ARM}] {len(arm_resps)} captions found...")
        scored_records = []

        def process_one(resp):
            sid = str(resp["sample_id"])
            cap = resp.get("caption", "none")
            tgt_ref = tgt_refs.get(sid, "")
            kw_list = keywords.get(sid, [])

            if cap == "none" or not tgt_ref:
                sim = 0.0
            else:
                sim = scorer.score(cap, tgt_ref)

            matched_gpt = matcher.match(kw_list, cap)
            kmr_val = (len(matched_gpt) / len(kw_list)) if kw_list else 0.0

            return {
                "arm": ARM,
                "victim": v_name,
                "sample_id": sid,
                "caption": cap,
                "target_keywords": str(kw_list),
                "asr_scores": f"[{sim}]",
                "max_score": sim,
                "mean_score": sim,
                "is_success": (sim >= thr),
                "ref_type": "target_victim",
                "matched_keywords_gpt": str(matched_gpt),
                "kmr_gpt": round(kmr_val, 4)
            }

        with cf.ThreadPoolExecutor(max_workers=10) as exc:
            futures = [exc.submit(process_one, r) for r in arm_resps]
            for fut in cf.as_completed(futures):
                scored_records.append(fut.result())

        scored_records.sort(key=lambda x: int(x["sample_id"]))

        suc_count = sum(1 for r in scored_records if r["is_success"])
        total_count = len(scored_records) if scored_records else 1
        asr = suc_count / total_count * 100.0
        mean_sim = np.mean([r["mean_score"] for r in scored_records]) if scored_records else 0.0
        avg_kmr = np.mean([r["kmr_gpt"] for r in scored_records]) * 100.0 if scored_records else 0.0

        print(f"  ==> [{v_name}|{ARM}] Target-ASR: {asr:5.1f}% | Official GPT-KMR: {avg_kmr:5.2f}% | Mean Sim: {mean_sim:.4f}")
        summary_rows.append({
            "Victim": v_name,
            "Arm": ARM,
            "Target-ASR(%)": round(asr, 1),
            "Official_GPT_KMR(%)": round(avg_kmr, 2),
            "Mean_Sim": round(mean_sim, 4)
        })

        if csv_p.exists():
            df = pd.read_csv(csv_p)
            df = df[df["arm"] != ARM]
            df_new = pd.DataFrame(scored_records)
            df_combined = pd.concat([df, df_new], ignore_index=True)
            df_combined.to_csv(csv_p, index=False)
            print(f"  [Disk] Saved updated {csv_p.name} with {len(df_combined)} rows.")

    summary_df = pd.DataFrame(summary_rows)
    md_out = ROOT / "outputs" / f"benchmark_{ARM}_summary.md"
    md_content = f"# Benchmark Summary: {ARM}\n\n" + summary_df.to_markdown(index=False)
    md_out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved summary to {md_out}")
    print("\n" + summary_df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description=f"Benchmark pipeline for {ARM}")
    parser.add_argument("--generate", action="store_true", help="Generate 100 adversarial images")
    parser.add_argument("--eval-commercial", action="store_true", help="Query commercial APIs")
    parser.add_argument("--eval-opensource", action="store_true", help="Query LLaVA-1.5-7B")
    parser.add_argument("--score", action="store_true", help="Score dual metrics")
    parser.add_argument("--all", action="store_true", help="Run full end-to-end pipeline")
    args = parser.parse_args()

    if not any([args.generate, args.eval_commercial, args.eval_opensource, args.score, args.all]):
        args.all = True

    if args.all or args.generate:
        generate_adv_images()

    if args.all or args.eval_commercial:
        run_commercial_inference()

    if args.all or args.eval_opensource:
        run_opensource_inference()

    if args.all or args.score:
        score_dual_metrics()


if __name__ == "__main__":
    main()
