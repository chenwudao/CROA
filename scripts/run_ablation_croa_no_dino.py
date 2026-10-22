"""Ablation pipeline: CROA without DINOv2 (w/o DINO).

Strictly controlled comparison against Full CROA (Candidate B):
  - 4-pass MCA, scale range [0.35, 1.0]
  - CLIP ensemble (B16, B32, Laion) with local OT weight 0.2
  - Active Source Repulsion (beta_rep = 0.5)
  - Mean-normalized momentum (mu = 0.9), 300 steps, eps = 16/255
  - NO DINOv2 branch (isolates DINOv2 dense geometric representation)

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
from croa.common import _kmeans, _norm255, _sinkhorn_ot, pil_to255
from scripts.eval_official_gpt_kmr import GPTKeywordMatcher
from croa.models import FeatureBank, make_victim
from scripts.run_benchmark import GPTScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARM = "croa_no_dino"
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


def rrc_crop_custom(x255: torch.Tensor, scale_range: tuple[float, float], gen: torch.Generator) -> torch.Tensor:
    _, _, h, w = x255.shape
    area = h * w
    target_area = area * (scale_range[0] + float(torch.rand(1, generator=gen)) * (scale_range[1] - scale_range[0]))
    aspect = w / h
    cw = max(1, min(w, int(round(math.sqrt(target_area * aspect)))))
    ch = max(1, min(h, int(round(math.sqrt(target_area / aspect)))))
    top = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=gen))
    left = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=gen))
    return x255[:, :, top:top + ch, left:left + cw]


def run_croa_no_dino(bank: FeatureBank, source_pil: Image.Image, target_pil: Image.Image,
                     passes: int = 4, scale_range: tuple[float, float] = (0.35, 1.0),
                     beta_rep: float = 0.5, cluster_number: int = 10,
                     clip_local_weight: float = 0.2,
                     steps: int = 300, seed: int = 20260827,
                     alpha: float = 1.0, epsilon: float = 16.0,
                     momentum_decay: float = 0.9) -> tuple[torch.Tensor, float, float]:
    """CROA without DINOv2 branch (CLIP ensemble + 4-crop MCA + Repulsion + Momentum)."""
    x0 = pil_to255(source_pil).to(DEVICE)
    xt = pil_to255(target_pil).to(DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gen_km = torch.Generator(device="cpu").manual_seed(seed + 1)
    delta = torch.zeros_like(x0)
    momentum = torch.zeros_like(x0)

    keys = ["B16", "B32", "Laion"]

    def kmeans_tokens(tokens: torch.Tensor) -> torch.Tensor:
        return _kmeans(tokens, cluster_number, gen_km)

    with torch.no_grad():
        s_px = _norm255(x0, C.ATTACK_RES)
        src_gl = {}
        for k in keys:
            m = bank.clips[k].vision_model
            h = m(pixel_values=s_px).last_hidden_state.float()
            src_gl[k] = (F.normalize(h[:, 0], dim=-1), F.normalize(h[:, 1:], dim=-1))

    final_loss = 0.0
    for _ in range(steps):
        delta.requires_grad_(True)
        adv_crops = [_norm255(rrc_crop_custom(x0 + delta, scale_range, gen), C.ATTACK_RES) for _ in range(passes)]
        tgt_crops = [_norm255(rrc_crop_custom(xt, scale_range, gen), C.ATTACK_RES) for _ in range(passes)]
        adv_px = torch.cat(adv_crops, dim=0)
        tgt_px = torch.cat(tgt_crops, dim=0)

        adv_gl = {}
        tgt_gl = {}
        for k in keys:
            m = bank.clips[k].vision_model
            h_a = m(pixel_values=adv_px).last_hidden_state.float()
            adv_gl[k] = (F.normalize(h_a[:, 0], dim=-1), F.normalize(h_a[:, 1:], dim=-1))
            with torch.no_grad():
                h_t = m(pixel_values=tgt_px).last_hidden_state.float()
                tgt_gl[k] = (F.normalize(h_t[:, 0], dim=-1), F.normalize(h_t[:, 1:], dim=-1))

        crop_losses = []
        for i in range(passes):
            clip_tgt_losses = []
            clip_rep_losses = []
            for k in keys:
                a_g = adv_gl[k][0][i:i+1]
                t_g = tgt_gl[k][0][i:i+1]
                s_g = src_gl[k][0]
                g_loss = _sinkhorn_ot(a_g, t_g.squeeze(0), eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS, tol=C.FOA_SINKHORN_TOL)
                l_loss = _sinkhorn_ot(kmeans_tokens(adv_gl[k][1][i]), kmeans_tokens(tgt_gl[k][1][i]),
                                      eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS, tol=C.FOA_SINKHORN_TOL)
                rep_loss = torch.sum(a_g * s_g, dim=-1).mean()
                clip_tgt_losses.append(g_loss + clip_local_weight * l_loss)
                clip_rep_losses.append(rep_loss)

            loss_clip = sum(clip_tgt_losses[j] - beta_rep * clip_rep_losses[j]
                            for j in range(len(keys))) / len(keys)
            crop_losses.append(loss_clip)

        loss_unified = sum(crop_losses) / len(crop_losses)
        final_loss = float(loss_unified.detach())

        grad = torch.autograd.grad(loss_unified, delta, create_graph=False)[0]
        with torch.no_grad():
            grad_norm = grad / (torch.mean(torch.abs(grad)) + 1e-8)
            momentum = momentum_decay * momentum + grad_norm
            delta = torch.clamp(delta + alpha * momentum.sign(), -epsilon, epsilon)

    linf = float(delta.detach().abs().max()) / 255.0
    adv = torch.clamp((x0 + delta.detach()) / 255.0, 0.0, 1.0)
    return adv[0], linf, final_loss


def generate_adv_images():
    print("\n" + "=" * 60)
    print(f"STAGE 1: Generating 100 Adversarial Images for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)
    bank = FeatureBank(device=DEVICE)

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

        t0 = time.time()
        adv_tensor, linf, final_loss = run_croa_no_dino(
            bank=bank,
            source_pil=src_pil,
            target_pil=tgt_pil,
            passes=4,
            scale_range=(0.35, 1.0),
            beta_rep=0.5,
            cluster_number=10,
            clip_local_weight=0.2,
            steps=300
        )
        elapsed = time.time() - t0

        adv_arr = (adv_tensor.cpu().permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
        Image.fromarray(adv_arr).save(out_png)

        rec = {
            "arm": ARM,
            "sample_id": sid,
            "loss": final_loss,
            "linf": linf,
            "elapsed_sec": elapsed,
            "timestamp": time.time(),
            "config": {
                "passes": 4,
                "scale_range": [0.35, 1.0],
                "dino": False,
                "source_repulsion": True,
                "beta_rep": 0.5,
                "clip_local_weight": 0.2,
                "steps": 300,
                "momentum": 0.9
            }
        }
        with REC_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"[{i+1}/100] Pair {sid:02d} | loss={final_loss:.4f} | linf={linf:.4f} | time={elapsed:.1f}s", flush=True)

    print(f"\n[Generation Complete] 100 images saved in {ADV_DIR} (total time: {time.time()-t0_all:.1f}s)")
    del bank
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
    md_out = ROOT / "outputs" / f"ablation_{ARM}_summary.md"
    md_content = f"# Ablation Summary: {ARM}\n\n" + summary_df.to_markdown(index=False)
    md_out.write_text(md_content, encoding="utf-8")
    print(f"\nSaved summary to {md_out}")
    print("\n" + summary_df.to_markdown(index=False))


def main():
    parser = argparse.ArgumentParser(description=f"Ablation pipeline for {ARM}")
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
