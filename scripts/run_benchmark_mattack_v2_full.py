"""Benchmark pipeline: Full Official M-Attack-V2 (arXiv:2602.17645).

Full reproduction of M-Attack-V2 with all four core components:
  1. MCA (Multi-Crop Alignment): 4-pass multi-crop on source adversarial image (scale [0.5, 1.0])
  2. ATA (Auxiliary Target Alignment): Small auxiliary target set from data/retrieved_embeddings/{sid}/
     (target_num = 3: 1 primary + 2 retrieved targets, weighted by beta = 0.3)
  3. PE+ (Patch-size Ensemble): 4 CLIP models with varying patch sizes:
     - ViT-B/16 (patch size 16)
     - ViT-B/32 (patch size 32)
     - ViT-B/32 LAION (patch size 32)
     - ViT-G/14 LAION (patch size 14)
  4. Patch Momentum: Adam optimizer on delta (lr = 0.005, betas=(0.9, 0.999), eps_clamp = 16/255)

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
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import CLIPModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from croa import config as C
from scripts.eval_official_gpt_kmr import GPTKeywordMatcher
from croa.models import make_victim
from scripts.run_benchmark import GPTScorer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARM = "mattack_v2_full"
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

RETRIEVAL_DIR = ROOT / "data" / "retrieved_embeddings"

def _resolve_model_path(hub_id: str, local_dir_name: str) -> str:
    p = ROOT / "models_cache" / local_dir_name
    if (p / "config.json").exists():
        return str(p)
    return hub_id

PE_PLUS_MODELS = {
    "clip_b16": _resolve_model_path("openai/clip-vit-base-patch16", "clip-vit-base-patch16"),
    "clip_b32": _resolve_model_path("openai/clip-vit-base-patch32", "clip-vit-base-patch32"),
    "clip_laion_b32": _resolve_model_path("laion/CLIP-ViT-B-32-laion2B-s34B-b79K", "CLIP-ViT-B-32-laion2B-s34B-b79K"),
    "clip_laion_g14": _resolve_model_path("laion/CLIP-ViT-G-14-laion2B-s12B-b42K", "laion-clip-vit-g-14"),
}

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=DEVICE).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=DEVICE).view(1, 3, 1, 1)


class PEPlusFeatureBank:
    """Refined Patch-size Ensemble (PE+) with 4 CLIP backbones."""
    def __init__(self, device: str = DEVICE):
        self.device = device
        self.models = {}
        for name, model_path in PE_PLUS_MODELS.items():
            print(f"Loading PE+ backbone: {name} ({model_path})...", flush=True)
            m = CLIPModel.from_pretrained(model_path).to(device).eval()
            for p in m.parameters():
                p.requires_grad = False
            self.models[name] = m
        print("All 4 PE+ backbones loaded successfully.", flush=True)

    def extract_features(self, px_norm: torch.Tensor) -> dict[str, torch.Tensor]:
        """Extract L2-normalized projected CLS features for each backbone."""
        feats = {}
        for name, m in self.models.items():
            f = m.get_image_features(pixel_values=px_norm)
            feats[name] = F.normalize(f.float(), dim=-1)
        return feats


def norm_clip(x: torch.Tensor, size: int = 224) -> torch.Tensor:
    """Resize and normalize with CLIP stats. Input in [0, 1]."""
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    x = torch.clamp(x, 0.0, 1.0)
    return (x - CLIP_MEAN) / CLIP_STD


def run_mattack_v2_full(bank: PEPlusFeatureBank,
                        source_pil: Image.Image,
                        primary_tgt_pil: Image.Image,
                        aux_tgt_pils: list[Image.Image],
                        steps: int = 300,
                        multi_pass_num: int = 4,
                        beta: float = 0.3,
                        alpha: float = 0.005,
                        epsilon: float = 16.0 / 255.0,
                        seed: int = 20260827) -> tuple[torch.Tensor, float, float]:
    """Full M-Attack-V2: MCA + ATA + PE+ + Adam Optimizer."""
    torch.manual_seed(seed)
    x0 = transforms.ToTensor()(source_pil.convert("RGB")).unsqueeze(0).to(DEVICE)
    
    # Target transformations
    target_crop_fn = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=[0.95, 1.0]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
    ])
    
    # Source MCA crop
    source_crop_fn = transforms.RandomResizedCrop(224, scale=[0.5, 1.0])
    
    # Prepare all target images: [primary, aux1, aux2, ...]
    all_tgts = [primary_tgt_pil] + aux_tgt_pils[:2]
    # Pad with primary target if fewer than 3
    while len(all_tgts) < 3:
        all_tgts.append(primary_tgt_pil)
    
    tgt_tensors = [transforms.ToTensor()(img.convert("RGB")).unsqueeze(0).to(DEVICE) for img in all_tgts]

    # Delta initialization & Adam optimizer
    delta = torch.zeros_like(x0, requires_grad=True)
    optimizer = torch.optim.Adam([delta], lr=alpha, betas=(0.9, 0.999))

    final_loss = 0.0

    for step in range(steps):
        optimizer.zero_grad()
        
        # Step-wise target feature extraction with ATA transform under no_grad
        with torch.no_grad():
            tgt_features_list = []
            for t_idx, tgt_tensor in enumerate(tgt_tensors):
                t_cropped = target_crop_fn(tgt_tensor)
                t_norm = norm_clip(t_cropped, 224)
                f_tgt = bank.extract_features(t_norm)
                tgt_features_list.append(f_tgt)

        # Multi-pass MCA on adversarial image
        accum_loss = 0.0
        for pass_idx in range(multi_pass_num):
            adv = torch.clamp(x0 + delta, 0.0, 1.0)
            adv_crop = source_crop_fn(adv)
            adv_norm = norm_clip(adv_crop, 224)
            adv_feats = bank.extract_features(adv_norm)

            # ATA Loss: Primary target (weight 1.0) + Auxiliary targets (weight beta=0.3)
            # Loss = -(Sim(adv, primary) + beta * Sum(Sim(adv, aux_j))) / |models|
            pass_loss = 0.0
            for name in bank.models:
                sim_primary = torch.sum(adv_feats[name] * tgt_features_list[0][name], dim=-1).mean()
                sim_aux = sum(torch.sum(adv_feats[name] * tgt_features_list[j][name], dim=-1).mean()
                              for j in range(1, len(tgt_features_list)))
                pass_loss += (sim_primary + beta * sim_aux) / len(bank.models)

            # Negate because optimizer minimizes loss to maximize similarity
            obj = -pass_loss / multi_pass_num
            obj.backward()
            accum_loss += float(pass_loss.detach()) / multi_pass_num

        final_loss = accum_loss
        optimizer.step()

        # Projection clamp
        with torch.no_grad():
            delta.clamp_(-epsilon, epsilon)

    linf = float(delta.detach().abs().max())
    adv_final = torch.clamp(x0 + delta.detach(), 0.0, 1.0)
    return adv_final[0], linf, final_loss


def generate_adv_images():
    print("\n" + "=" * 60)
    print(f"STAGE 1: Generating 100 Adversarial Images for '{ARM}'")
    print("=" * 60)
    sys.stdout.flush()

    pairs = C.load_pairs(100)
    bank = PEPlusFeatureBank(device=DEVICE)

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

        # Load auxiliary retrieved target images for ATA
        aux_pils = []
        ret_sub = RETRIEVAL_DIR / str(sid)
        if ret_sub.exists():
            for img_f in sorted(ret_sub.glob("*.jpg")):
                try:
                    aux_pils.append(Image.open(img_f).convert("RGB"))
                except Exception:
                    pass

        t0 = time.time()
        adv_tensor, linf, final_loss = run_mattack_v2_full(
            bank=bank,
            source_pil=src_pil,
            primary_tgt_pil=tgt_pil,
            aux_tgt_pils=aux_pils,
            steps=300,
            multi_pass_num=4,
            beta=0.3,
            alpha=0.005,
            epsilon=16.0 / 255.0
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
                "mca_passes": 4,
                "ata_target_num": 3,
                "ata_beta": 0.3,
                "pe_plus_backbones": list(PE_PLUS_MODELS.keys()),
                "optimizer": "adam",
                "alpha": 0.005,
                "steps": 300,
                "eps": 16.0 / 255.0
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
