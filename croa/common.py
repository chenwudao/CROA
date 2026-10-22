"""Shared tensor ops for the FOA-Attack v2 benchmark.

Ports of the official preprocessing / augmentation / Sinkhorn-OT utilities,
verified line-by-line against the authors' released code (2026-09-10):
  - _crop_official: MPCAttack official aspect-preserving random crop + resize-back
  - _rrc_crop:      M-Attack / FOA-Attack official RandomResizedCrop (crop only,
                    locked original aspect ratio; resize happens in _norm255)
  - _sinkhorn_ot / _kmeans: FOA-Attack official entropic OT and kmeans_pytorch
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from PIL import Image

CLIP_STATS_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STATS_STD = (0.26862954, 0.26130258, 0.27577711)

RRC_SCALE = (0.5, 0.9)          # official crop_scale, both M-Attack and FOA-Attack
RRC_RATIO = (3 / 4, 4 / 3)      # torchvision RandomResizedCrop default ratio
SINKHORN_EPS = 0.1
SINKHORN_ITERS = 100


def unwrap_feats(x):
    """transformers v5 get_image_features/get_text_features return output objects
    with the projected features in .pooler_output; v4 returned plain tensors."""
    if torch.is_tensor(x):
        return x
    return x.pooler_output


def pil_to255(pil: Image.Image) -> torch.Tensor:
    """PIL -> [1,3,H,W] float tensor in the 0-255 domain (official MPC convention)."""
    import numpy as np
    return torch.from_numpy(np.array(pil.convert("RGB"))).permute(2, 0, 1).unsqueeze(0).float()


def _norm255(x255: torch.Tensor, size: int) -> torch.Tensor:
    """Resize to `size` and normalise with CLIP stats (official both-repo path)."""
    x = F.interpolate(x255, size=(size, size), mode="bicubic", align_corners=False)
    x = torch.clamp(x, 0.0, 255.0) / 255.0
    m = torch.tensor(CLIP_STATS_MEAN, device=x.device).view(1, 3, 1, 1)
    s = torch.tensor(CLIP_STATS_STD, device=x.device).view(1, 3, 1, 1)
    return (x - m) / s


def _crop_official(x: torch.Tensor, scale: tuple[float, float],
                   gen: torch.Generator) -> torch.Tensor:
    """MPC official per-step augmentation: aspect-preserving random crop, resized back."""
    _, _, h, w = x.shape
    area = h * w * (scale[0] + (scale[1] - scale[0]) * torch.rand(1, generator=gen).item())
    aspect = w / h
    cw = max(1, min(w, int(round((area * aspect) ** 0.5))))
    ch = max(1, min(h, int(round((area / aspect) ** 0.5))))
    y0 = torch.randint(0, h - ch + 1, (1,), generator=gen).item()
    x0 = torch.randint(0, w - cw + 1, (1,), generator=gen).item()
    crop = x[:, :, y0:y0 + ch, x0:x0 + cw]
    return F.interpolate(crop, size=(h, w), mode="bilinear", align_corners=False)


def _rrc_crop(x255: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """M-Attack / FOA-Attack official crop with locked original aspect ratio (crop
    only; the resize to 224 happens inside _norm255)."""
    _, _, h, w = x255.shape
    area = h * w
    target_area = area * (RRC_SCALE[0] + float(torch.rand(1, generator=gen)) * (RRC_SCALE[1] - RRC_SCALE[0]))
    aspect = w / h  # locked original aspect ratio, strictly adhering to author's script
    cw = max(1, min(w, int(round(math.sqrt(target_area * aspect)))))
    ch = max(1, min(h, int(round(math.sqrt(target_area / aspect)))))
    top = int(torch.randint(0, max(1, h - ch + 1), (1,), generator=gen))
    left = int(torch.randint(0, max(1, w - cw + 1), (1,), generator=gen))
    return x255[:, :, top:top + ch, left:left + cw]


def ot_matched_sim(a_tok: torch.Tensor, b_tok: torch.Tensor) -> torch.Tensor:
    """Sinkhorn-OT matched cosine between two token sets [N,d], [M,d].
    Plan computed without gradient (FOA convention); gradients flow through
    the matched similarity only."""
    sim = a_tok @ b_tok.T
    K = torch.exp(-(1.0 - sim.detach()) / SINKHORN_EPS)
    n, m = K.shape
    u = torch.full((n,), 1.0 / n, device=sim.device)
    v = torch.full((m,), 1.0 / m, device=sim.device)
    r, c = torch.ones_like(u), torch.ones_like(v)
    with torch.no_grad():
        for _ in range(SINKHORN_ITERS):
            r = u / (K @ c)
            c = v / (K.T @ r)
        plan = torch.outer(r, c) * K
    return (plan * sim).sum()


def _sinkhorn_ot(src: torch.Tensor, tgt: torch.Tensor,
                 eps: float = 0.1, iters: int = 100, tol: float = 1e-2) -> torch.Tensor:
    """Official FOA OT: Gibbs kernel on 1-cos, uniform marginals, early stop."""
    if src.dim() == 1:
        src = src.unsqueeze(0)
    elif src.dim() == 3 and src.shape[0] == 1:
        src = src.squeeze(0)
    if tgt.dim() == 1:
        tgt = tgt.unsqueeze(0)
    elif tgt.dim() == 3 and tgt.shape[0] == 1:
        tgt = tgt.squeeze(0)
    sim = src @ tgt.t()                                 # grad flows through sim
    K = torch.exp(-(1.0 - sim.detach()) / eps)
    n, m = K.shape
    u = torch.full((n,), 1.0 / n, device=sim.device)
    v = torch.full((m,), 1.0 / m, device=sim.device)
    r, c = torch.ones_like(u), torch.ones_like(v)
    with torch.no_grad():
        for _ in range(iters):
            r0 = r
            r = u / torch.clamp(K @ c, min=1e-8)
            c = v / torch.clamp(K.t() @ r, min=1e-8)
            if (r - r0).abs().mean() < tol:
                break
    plan = torch.outer(r, c) * K
    return (plan * sim).sum()


def _kmeans(tokens: torch.Tensor, k: int, gen: torch.Generator,
            iters: int = 100) -> torch.Tensor:
    """kmeans_pytorch-equivalent: euclidean, random-choice init, iter_limit=100.
    tokens: [N, d] or [1, N, d] L2-normed patch tokens. Returns [k, d] centroids."""
    if tokens.dim() == 3 and tokens.shape[0] == 1:
        tokens = tokens.squeeze(0)
    elif tokens.dim() == 1:
        tokens = tokens.unsqueeze(0)
    idx = torch.randperm(tokens.shape[0], generator=gen)[:k]
    centers = tokens[idx].clone()
    for _ in range(iters):
        assign = (tokens @ centers.t()).argmax(dim=-1)             # [N]
        counts = torch.bincount(assign, minlength=k).clamp(min=1).unsqueeze(-1)
        sums = torch.zeros_like(centers).index_add_(0, assign, tokens)
        new_centers = sums / counts
        if torch.allclose(new_centers, centers, atol=1e-6):
            break
        centers = new_centers
    return centers

