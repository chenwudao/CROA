"""Pipeline to run commercial model inference and dual-metric evaluation for CROA-Tuned.

Victims:
- gpt-5
- gemini-3.5-flash-lite
- claude-4.5-haiku

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
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from croa import config as C
from scripts.eval_official_gpt_kmr import GPTKeywordMatcher
from croa.models import make_victim
from scripts.run_benchmark import GPTScorer

VICTIMS = ["gpt-5", "gemini-3.5-flash-lite", "claude-4.5-haiku"]
ARM = "croa_tuned"
ADV_DIR = ROOT / "outputs" / "adv" / ARM
RESP_DIR = ROOT / "outputs" / "responses"
DESC_DIR = ROOT / "outputs" / "descriptions"


def backup_scored_csvs():
    for v in VICTIMS:
        csv_p = RESP_DIR / f"scored_{v}.csv"
        if csv_p.exists():
            bak_p = RESP_DIR / f"scored_{v}.csv.bak_croa"
            if not bak_p.exists():
                shutil.copy2(csv_p, bak_p)
                print(f"[Backup] Created backup: {bak_p.name}")


def run_commercial_inference():
    print("\n" + "=" * 60)
    print(f"PHASE 1: Commercial Inference for ARM '{ARM}' (100 pairs)")
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
                        if r.get("arm") == ARM:
                            done.add(str(r["sample_id"]))
                    except Exception:
                        pass

        missing = [p for p in pairs if str(p["sample_id"]) not in done]
        print(f"\n[Victim: {v_name}] {len(done)}/100 already completed, {len(missing)} remaining.")
        sys.stdout.flush()

        if not missing:
            continue

        victim = make_victim(v_name)
        with resp_file.open("a", encoding="utf-8") as fp:
            for i, p in enumerate(missing):
                sid = str(p["sample_id"])
                adv_png = ADV_DIR / f"{sid}.png"
                if not adv_png.exists():
                    print(f"  [ERROR] Missing {adv_png}! Ensure generation has completed.")
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
                    "caption": cap
                }
                fp.write(json.dumps(entry, ensure_ascii=False) + "\n")
                fp.flush()
                print(f"  [{v_name}|{ARM}] {sid} ({i+1}/{len(missing)}): {cap[:60]}...", flush=True)

        del victim
        if hasattr(Image, "close"):
            pass


def run_target_asr_scoring():
    print("\n" + "=" * 60)
    print(f"PHASE 2: Target-ASR Scoring for ARM '{ARM}'")
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

        # Load target refs
        tgt_refs = {}
        for line in target_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    tgt_refs[str(r["sample_id"])] = r["caption"]
                except Exception:
                    pass

        # Load responses for this arm
        arm_resps = []
        for line in resp_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                    if r.get("arm") == ARM:
                        arm_resps.append(r)
                except Exception:
                    pass

        print(f"\n[Scoring Target-ASR: {v_name}] {len(arm_resps)} adversarial captions...")
        sys.stdout.flush()

        scored_records = []
        for r in arm_resps:
            sid = str(r["sample_id"])
            cap = r["caption"]
            ref_cap = tgt_refs.get(sid, "")
            score = scorer.score(cap, ref_cap) if ref_cap and cap != "none" else 0.0
            is_succ = (score >= thr)
            scored_records.append({
                "arm": ARM,
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
        asr = np.mean([1.0 if s["is_success"] else 0.0 for s in scored_records]) * 100.0
        mean_sim = np.mean([s["max_score"] for s in scored_records])
        print(f"  ==> {v_name} Target-ASR: {asr:.2f}% | Mean Sim: {mean_sim:.4f}")
        sys.stdout.flush()

    return results


def run_official_gpt_kmr_scoring(asr_results):
    print("\n" + "=" * 60)
    print(f"PHASE 3: Official GPT-KMR Scoring for ARM '{ARM}'")
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
        kmr = np.mean([s["kmr_gpt"] for s in scored]) * 100.0
        print(f"  ==> {v_name} Official GPT-KMR: {kmr:.2f}%")
        sys.stdout.flush()

    return final_records_by_victim


def merge_into_scored_csvs(final_records_by_victim):
    print("\n" + "=" * 60)
    print(f"PHASE 4: Merging '{ARM}' into outputs/responses/scored_*.csv")
    print("=" * 60)
    sys.stdout.flush()

    cols = ['arm', 'victim', 'sample_id', 'caption', 'target_keywords',
            'asr_scores', 'max_score', 'mean_score', 'is_success',
            'ref_type', 'matched_keywords_gpt', 'kmr_gpt']

    summary_data = []

    for v_name in VICTIMS:
        csv_p = RESP_DIR / f"scored_{v_name}.csv"
        new_records = final_records_by_victim.get(v_name, [])
        if not new_records:
            continue

        df_new = pd.DataFrame(new_records)[cols]

        if csv_p.exists():
            df_existing = pd.read_csv(csv_p)
            # Remove any previous records for this ARM
            df_existing = df_existing[df_existing["arm"] != ARM]
            df_merged = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_merged = df_new

        df_merged.to_csv(csv_p, index=False)
        print(f"Saved {len(df_merged)} rows to {csv_p.name} (includes {len(df_new)} '{ARM}' rows).")

        # Compute summary metrics for report
        g_tuned = df_new
        asr_tuned = float(g_tuned["is_success"].mean() * 100)
        kmr_tuned = float(g_tuned["kmr_gpt"].mean() * 100)
        sim_tuned = float(g_tuned["max_score"].mean())

        # Compare with baseline foa_v2
        df_base = pd.read_csv(csv_p)
        g_base = df_base[df_base["arm"] == "foa_v2"]
        asr_base = float(g_base["is_success"].mean() * 100) if len(g_base) > 0 else 0.0
        kmr_base = float(g_base["kmr_gpt"].mean() * 100) if len(g_base) > 0 else 0.0
        sim_base = float(g_base["max_score"].mean()) if len(g_base) > 0 else 0.0

        summary_data.append({
            "Victim": v_name,
            "CROA_Base_ASR(%)": asr_base,
            "CROA_Tuned_ASR(%)": asr_tuned,
            "Delta_ASR(%)": round(asr_tuned - asr_base, 2),
            "CROA_Base_KMR(%)": kmr_base,
            "CROA_Tuned_KMR(%)": kmr_tuned,
            "Delta_KMR(%)": round(kmr_tuned - kmr_base, 2),
            "CROA_Base_Sim": round(sim_base, 3),
            "CROA_Tuned_Sim": round(sim_tuned, 3),
        })

    sum_df = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print("HEAD-TO-HEAD COMPARISON: CROA Baseline vs CROA-Tuned (Candidate B)")
    print("=" * 80)
    print(sum_df.to_string(index=False))

    comp_file = ROOT / "outputs" / "croa_tuned_comparison.md"
    with open(comp_file, "w", encoding="utf-8") as f:
        f.write("# CROA Baseline vs CROA-Tuned (Candidate B) Head-to-Head Comparison\n\n")
        f.write("Candidate B Configuration: 4-pass MCA, scale (0.35, 1.0), beta_rep=0.5, Kc=10, 300 steps\n\n")
        f.write(sum_df.to_markdown(index=False) + "\n")
    print(f"\nSaved comparison markdown to {comp_file}")


def main():
    backup_scored_csvs()
    run_commercial_inference()
    asr_results = run_target_asr_scoring()
    final_records = run_official_gpt_kmr_scoring(asr_results)
    merge_into_scored_csvs(final_records)
    print("\n[Done] All commercial evaluations for CROA-Tuned completed successfully!")


if __name__ == "__main__":
    main()
