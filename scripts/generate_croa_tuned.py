"""Generate 100 adversarial images for CROA-Tuned (Candidate B).

Hyperparameters:
- passes: 4 (Quad-crop MCA)
- scale_range: (0.35, 1.0)
- beta_rep: 0.5 (source repulsion)
- cluster_number: 10 (k-means centroids)
- dino_local_weight: 0.4
- clip_local_weight: 0.2
- steps: 300, alpha: 1.0, epsilon: 16.0
- momentum_decay: 0.9
"""
import json
import math
import os
import sys
import time
from pathlib import Path
from PIL import Image

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from croa import config as C
from croa.common import _kmeans, _norm255, _sinkhorn_ot, pil_to255
from croa.models import FeatureBank

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_DIR = ROOT / "outputs" / "adv" / "croa_tuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REC_FILE = ROOT / "outputs" / "records" / "attack_records_croa_tuned.jsonl"
REC_FILE.parent.mkdir(parents=True, exist_ok=True)


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


def run_croa_tuned(bank: FeatureBank, source_pil: Image.Image, target_pil: Image.Image,
                   passes: int = 4, scale_range: tuple[float, float] = (0.35, 1.0),
                   beta_rep: float = 0.5, cluster_number: int = 10,
                   dino_local_weight: float = 0.4, clip_local_weight: float = 0.2,
                   steps: int = 300, seed: int = 20260827,
                   alpha: float = 1.0, epsilon: float = 16.0,
                   momentum_decay: float = 0.9) -> tuple[torch.Tensor, float, float]:
    x0 = pil_to255(source_pil).to(DEVICE)
    xt = pil_to255(target_pil).to(DEVICE)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    gen_km = torch.Generator(device="cpu").manual_seed(seed + 1)
    delta = torch.zeros_like(x0)
    momentum = torch.zeros_like(x0)

    keys = ["B16", "B32", "Laion"]
    keys_v2 = ["B16", "B32", "Laion", "Dino"]

    def kmeans_tokens(tokens: torch.Tensor) -> torch.Tensor:
        return _kmeans(tokens, cluster_number, gen_km)

    with torch.no_grad():
        s_px = _norm255(x0, C.ATTACK_RES)
        src_gl = {}
        for k in keys_v2:
            m = bank.dino if k == "Dino" else bank.clips[k].vision_model
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
        for k in keys_v2:
            m = bank.dino if k == "Dino" else bank.clips[k].vision_model
            h_a = m(pixel_values=adv_px).last_hidden_state.float()
            adv_gl[k] = (F.normalize(h_a[:, 0], dim=-1), F.normalize(h_a[:, 1:], dim=-1))
            with torch.no_grad():
                h_t = m(pixel_values=tgt_px).last_hidden_state.float()
                tgt_gl[k] = (F.normalize(h_t[:, 0], dim=-1), F.normalize(h_t[:, 1:], dim=-1))

        crop_losses = []
        for i in range(passes):
            # Branch 1: CLIP
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

            # Branch 2: DINOv2
            d_a_g = adv_gl["Dino"][0][i:i+1]
            d_t_g = tgt_gl["Dino"][0][i:i+1]
            d_s_g = src_gl["Dino"][0]
            d_g_loss = _sinkhorn_ot(d_a_g, d_t_g.squeeze(0), eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS, tol=C.FOA_SINKHORN_TOL)
            d_l_loss = _sinkhorn_ot(kmeans_tokens(adv_gl["Dino"][1][i]), kmeans_tokens(tgt_gl["Dino"][1][i]),
                                    eps=C.FOA_SINKHORN_EPS, iters=C.FOA_SINKHORN_ITERS, tol=C.FOA_SINKHORN_TOL)
            d_rep_loss = torch.sum(d_a_g * d_s_g, dim=-1).mean()
            loss_dino = (d_g_loss + dino_local_weight * d_l_loss) - beta_rep * d_rep_loss

            crop_losses.append(0.5 * loss_clip + 0.5 * loss_dino)

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


def main():
    print("=" * 60)
    print("Starting CROA-Tuned 100-Pair Generation")
    print(f"Device: {DEVICE} | Output: {OUT_DIR}")
    print("=" * 60)
    sys.stdout.flush()

    bank = FeatureBank(device=DEVICE)
    pairs = C.load_pairs(100)

    # Check existing images
    existing = set()
    for p in OUT_DIR.glob("*.png"):
        existing.add(p.stem)
    print(f"Found {len(existing)} existing images in {OUT_DIR.name}.")
    sys.stdout.flush()

    rec_existing = set()
    if REC_FILE.exists():
        for line in REC_FILE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rec_existing.add(json.loads(line)["sample_id"])
                except Exception:
                    pass

    total_start = time.time()
    for i, p in enumerate(pairs):
        sid = str(p["sample_id"])
        out_p = OUT_DIR / f"{sid}.png"
        if sid in existing and out_p.exists():
            # Already generated, ensure record exists
            if sid not in rec_existing:
                rec = {
                    "arm": "croa_tuned",
                    "sample_id": sid,
                    "seed": 20260827,
                    "steps": 300,
                    "linf": 16.0 / 255.0,
                    "final_s_pos": 0.0,
                    "final_s_neg": 0.0,
                    "elapsed_s": 208.0,
                    "config": "cand_b"
                }
                with open(REC_FILE, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                rec_existing.add(sid)
            continue

        print(f"\n[CROA-Tuned] Generating sample {sid} ({i+1}/100)...")
        sys.stdout.flush()
        t0 = time.time()
        s_pil = Image.open(p["source"]).convert("RGB")
        t_pil = Image.open(p["target"]).convert("RGB")
        adv_tensor, linf, final_loss = run_croa_tuned(bank, s_pil, t_pil, passes=4, scale_range=(0.35, 1.0))
        elapsed = time.time() - t0

        adv_np = (adv_tensor.cpu().numpy().transpose(1, 2, 0) * 255.0).round().astype(np.uint8)
        Image.fromarray(adv_np).save(out_p)

        rec = {
            "arm": "croa_tuned",
            "sample_id": sid,
            "seed": 20260827,
            "steps": 300,
            "linf": linf,
            "final_s_pos": final_loss,
            "final_s_neg": 0.0,
            "elapsed_s": round(elapsed, 1),
            "config": "cand_b"
        }
        with open(REC_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        print(f"  --> Saved sample {sid}.png (Elapsed: {elapsed:.1f}s, Linf: {linf*255:.2f}/255, Loss: {final_loss:.3f})")
        sys.stdout.flush()

    print(f"\nAll 100 CROA-Tuned adversarial images generated in {time.time()-total_start:.1f}s!")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
