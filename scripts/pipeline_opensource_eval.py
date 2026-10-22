"""Pipeline to run open-source model inference and dual-metric evaluation for CROA-Tuned, AnyAttack, and FOA-v2.

Victims:
- llava15-7b
- qwen2.5vl7b
- internvl3-8b

Arms:
- croa_tuned (Quad-crop MCA Candidate B)
- anyattack (Official feedforward generative CVPR 2025)
- foa_v2 (Dual-crop MCA baseline)

Metrics:
- Target-ASR (%) [GPTScore >= 0.5 via gpt-4o-mini]
- Official GPT-KMR (%) [VILA-Lab / FOA official judge via gpt-4o-mini]
"""
import concurrent.futures as cf
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from croa import config as C
from scripts.eval_official_gpt_kmr import GPTKeywordMatcher
from croa.models import make_victim
from scripts.run_benchmark import GPTScorer

VICTIMS = ["llava15-7b", "qwen2.5vl7b", "internvl3-8b"]
ARMS = ["croa_tuned", "anyattack", "foa_v2"]
ADV_ROOT = ROOT / "outputs" / "adv"
RESP_DIR = ROOT / "outputs" / "responses"
DESC_DIR = ROOT / "outputs" / "descriptions"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def backup_scored_csvs():
    for v in VICTIMS:
        csv_p = RESP_DIR / f"scored_{v}.csv"
        if csv_p.exists():
            bak_p = RESP_DIR / f"scored_{v}.csv.bak_croa"
            if not bak_p.exists():
                shutil.copy2(csv_p, bak_p)
                print(f"[Backup] Created backup: {bak_p.name}")


def run_opensource_inference():
    print("\n" + "=" * 60)
    print("PHASE 1: Open-Source Model Inference (100 pairs)")
    print(f"Arms: {ARMS} | Victims: {VICTIMS} | Device: {DEVICE}")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)
    RESP_DIR.mkdir(parents=True, exist_ok=True)

    for v_name in VICTIMS:
        resp_file = RESP_DIR / f"responses_{v_name}.jsonl"
        done = set()
        if resp_file.exists():
            for line in resp_file.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        r = json.loads(line)
                        done.add((r.get("arm"), str(r["sample_id"])))
                    except Exception:
                        pass

        # Check which arms have missing pairs
        needed = False
        for arm in ARMS:
            missing = [p for p in pairs if (arm, str(p["sample_id"])) not in done]
            if missing:
                needed = True
                break

        if not needed:
            print(f"\n[Victim: {v_name}] All target arms already completed.")
            continue

        print(f"\n>>> Loading victim model: {v_name} ({C.VICTIMS[v_name]}) ...")
        sys.stdout.flush()
        victim = make_victim(v_name, device=DEVICE)

        with resp_file.open("a", encoding="utf-8") as fp:
            for arm in ARMS:
                missing = [p for p in pairs if (arm, str(p["sample_id"])) not in done]
                print(f"  [{v_name}|{arm}] {len(pairs)-len(missing)}/100 done, {len(missing)} missing.")
                sys.stdout.flush()

                if not missing:
                    continue

                arm_dir = ADV_ROOT / arm
                t0 = time.time()
                for i, p in enumerate(missing):
                    sid = str(p["sample_id"])
                    adv_png = arm_dir / f"{sid}.png"
                    if not adv_png.exists():
                        print(f"    [MISSING] {adv_png}! Skipping.")
                        continue

                    adv_pil = Image.open(adv_png).convert("RGB")
                    try:
                        cap = victim.ask(adv_pil, C.CAPTION_PROMPT, max_new_tokens=32) or "none"
                    except Exception as e:
                        print(f"    [ERROR] {sid}: {e}")
                        cap = "none"

                    entry = {
                        "arm": arm,
                        "victim": v_name,
                        "sample_id": sid,
                        "caption": cap
                    }
                    fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    fp.flush()
                    if (i + 1) % 25 == 0 or (i + 1) == len(missing):
                        print(f"    [{v_name}|{arm}] {i+1}/{len(missing)} completed (elapsed {time.time()-t0:.1f}s)...", flush=True)

        del victim
        torch.cuda.empty_cache()
        time.sleep(2)


def run_target_asr_scoring():
    print("\n" + "=" * 60)
    print("PHASE 2: Target-ASR Scoring for Open-Source Victims")
    print("=" * 60)
    sys.stdout.flush()

    scorer = GPTScorer()
    thr = C.GPTSCORE_THRESHOLD

    results = {}
    for v_name in VICTIMS:
        resp_file = RESP_DIR / f"responses_{v_name}.jsonl"
        target_file = DESC_DIR / f"target_{v_name}.jsonl"

        if not resp_file.exists() or not target_file.exists():
            print(f"  [SKIP] Missing responses or targets for {v_name}")
            continue

        tgt_refs = {}
        for line in target_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    tgt_refs[str(r["sample_id"])] = r["caption"]
                except Exception:
                    pass

        raw_resps = []
        for line in resp_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("arm") in ARMS:
                        raw_resps.append(r)
                except Exception:
                    pass

        print(f"\n[Scoring Target-ASR: {v_name}] Total {len(raw_resps)} captions across {ARMS}...")
        sys.stdout.flush()

        scored_records = []
        for r in raw_resps:
            sid = str(r["sample_id"])
            cap = r["caption"]
            ref_cap = tgt_refs.get(sid, "")
            score = scorer.score(cap, ref_cap) if ref_cap and cap != "none" else 0.0
            is_succ = (score >= thr)
            scored_records.append({
                "arm": r["arm"],
                "victim": v_name,
                "sample_id": sid,
                "caption": cap,
                "asr_scores": [score],
                "max_score": round(score, 4),
                "mean_score": round(score, 4),
                "is_success": is_succ,
                "ref_type": "victim"
            })

        results[v_name] = scored_records

        df_v = pd.DataFrame(scored_records)
        for arm, g in df_v.groupby("arm"):
            asr = float(g["is_success"].mean() * 100)
            sim = float(g["max_score"].mean())
            print(f"  ==> [{v_name}|{arm:12s}] Target-ASR: {asr:5.1f}% | Mean Sim: {sim:.4f}")
        sys.stdout.flush()

    return results


def run_official_gpt_kmr_scoring(asr_results):
    print("\n" + "=" * 60)
    print("PHASE 3: Official GPT-KMR Scoring for Open-Source Victims")
    print("=" * 60)
    sys.stdout.flush()

    keywords = C.load_keywords()
    matcher = GPTKeywordMatcher()

    final_records_by_victim = {}

    for v_name in VICTIMS:
        records = asr_results.get(v_name, [])
        if not records:
            continue

        print(f"\n[Scoring Official GPT-KMR: {v_name}] {len(records)} records...")
        sys.stdout.flush()

        def process_row(row):
            sid = str(row["sample_id"])
            cap = str(row["caption"])
            tgt_kws = keywords.get(sid, [])
            matched_gpt = matcher.match(tgt_kws, cap)
            kmr_gpt = (len(matched_gpt) / len(tgt_kws)) if tgt_kws else 0.0
            return {
                "arm": row["arm"],
                "victim": row["victim"],
                "sample_id": row["sample_id"],
                "caption": row["caption"],
                "target_keywords": tgt_kws,
                "asr_scores": row["asr_scores"],
                "max_score": row["max_score"],
                "mean_score": row["mean_score"],
                "is_success": row["is_success"],
                "ref_type": row["ref_type"],
                "matched_keywords_gpt": matched_gpt,
                "kmr_gpt": round(kmr_gpt, 4)
            }

        with cf.ThreadPoolExecutor(max_workers=10) as ex:
            scored = list(ex.map(process_row, records))

        final_records_by_victim[v_name] = scored

        df_v = pd.DataFrame(scored)
        for arm, g in df_v.groupby("arm"):
            kmr = float(g["kmr_gpt"].mean() * 100)
            print(f"  ==> [{v_name}|{arm:12s}] Official GPT-KMR: {kmr:5.2f}%")
        sys.stdout.flush()

    return final_records_by_victim


def merge_into_scored_csvs(final_records_by_victim):
    print("\n" + "=" * 60)
    print("PHASE 4: Merging Open-Source Arms into outputs/responses/scored_*.csv")
    print("=" * 60)
    sys.stdout.flush()

    cols = ['arm', 'victim', 'sample_id', 'caption', 'target_keywords',
            'asr_scores', 'max_score', 'mean_score', 'is_success',
            'ref_type', 'matched_keywords_gpt', 'kmr_gpt']

    all_summary_rows = []

    for v_name in VICTIMS:
        csv_p = RESP_DIR / f"scored_{v_name}.csv"
        new_records = final_records_by_victim.get(v_name, [])
        if not new_records:
            continue

        df_new = pd.DataFrame(new_records)[cols]

        if csv_p.exists():
            df_existing = pd.read_csv(csv_p)
            # Remove any previous records for the evaluated ARMS
            df_existing = df_existing[~df_existing["arm"].isin(ARMS)]
            # Ensure all cols exist
            for col in cols:
                if col not in df_existing.columns:
                    df_existing[col] = "" if "keywords" in col or "ref" in col else (0.0 if "score" in col or "kmr" in col else False)
            df_merged = pd.concat([df_existing[cols], df_new], ignore_index=True)
        else:
            df_merged = df_new

        df_merged.to_csv(csv_p, index=False)
        print(f"Saved {len(df_merged)} rows to {csv_p.name} (includes {len(df_new)} updated rows for {ARMS}).")

        for arm, g in df_merged.groupby("arm", sort=False):
            asr = float(g["is_success"].mean() * 100)
            kmr = float(g["kmr_gpt"].mean() * 100) if "kmr_gpt" in g.columns else 0.0
            sim = float(g["max_score"].mean()) if "max_score" in g.columns else 0.0
            all_summary_rows.append({
                "Victim": v_name,
                "Arm": arm,
                "Target-ASR(%)": round(asr, 2),
                "Official_GPT_KMR(%)": round(kmr, 2),
                "Mean_Sim": round(sim, 3)
            })

    sum_df = pd.DataFrame(all_summary_rows)
    print("\n" + "=" * 80)
    print("OPEN-SOURCE FULL BENCHMARK LEADERBOARD (Table 2 View)")
    print("=" * 80)

    piv_asr = sum_df.pivot(index="Arm", columns="Victim", values="Target-ASR(%)")
    piv_kmr = sum_df.pivot(index="Arm", columns="Victim", values="Official_GPT_KMR(%)")

    # Add Average column
    piv_asr["Avg"] = piv_asr.mean(axis=1).round(2)
    piv_kmr["Avg"] = piv_kmr.mean(axis=1).round(2)

    print("\n=== Target-ASR (%) on Open-Source Models ===")
    print(piv_asr.to_markdown())

    print("\n=== Official GPT-KMR (%) on Open-Source Models ===")
    print(piv_kmr.to_markdown())

    out_md = ROOT / "outputs" / "opensource_comparison.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# Open-Source Benchmark Leaderboard (Table 2 Reproduction)\n\n")
        f.write("## 1. Target-ASR (%) Leaderboard [GPTScore >= 0.5]\n\n")
        f.write(piv_asr.to_markdown() + "\n\n")
        f.write("## 2. Official GPT-KMR (%) Leaderboard\n\n")
        f.write(piv_kmr.to_markdown() + "\n\n")
        f.write("## 3. Comprehensive Performance Matrix\n\n")
        f.write(sum_df.to_markdown(index=False) + "\n\n")
    print(f"\nSaved summary markdown to {out_md}")


def main():
    backup_scored_csvs()
    run_opensource_inference()
    asr_results = run_target_asr_scoring()
    final_records = run_official_gpt_kmr_scoring(asr_results)
    merge_into_scored_csvs(final_records)
    print("\n[Done] All open-source model evaluations completed successfully!")


if __name__ == "__main__":
    main()
