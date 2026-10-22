"""Repository smoke check.

CPU path (no model weights and no API key):

    python scripts/smoke_test.py

GPU path (CUDA plus the CLIP surrogate weights; two steps of the existing
M-Attack arm on pair 0):

    python scripts/smoke_test.py --gpu
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def check_layout() -> None:
    from croa import config as C

    if C.ROOT != REPO:
        raise SystemExit(f"config.ROOT is {C.ROOT}, expected {REPO}")
    pairs = C.load_pairs()
    if len(pairs) != C.NUM_PAIRS:
        raise SystemExit(f"expected {C.NUM_PAIRS} pairs, found {len(pairs)}")
    missing = [
        pair["sample_id"]
        for pair in pairs
        if not Path(pair["source"]).is_file() or not Path(pair["target"]).is_file()
    ]
    if missing:
        raise SystemExit(f"missing images for pairs: {', '.join(missing[:8])}")
    captions = C.load_captions()
    keywords = C.load_keywords()
    for i in range(C.NUM_PAIRS):
        if str(i) not in captions:
            raise SystemExit(f"missing captions for pair {i}")
        if str(i) not in keywords:
            raise SystemExit(f"missing keywords for pair {i}")
    print(f"data: {len(pairs)} pairs, captions and keywords ok")


def check_ot() -> None:
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("torch is not installed; skipped Sinkhorn/k-means check")
        return
    from croa.common import _kmeans, _sinkhorn_ot

    generator = torch.Generator(device="cpu").manual_seed(0)
    tokens = F.normalize(torch.randn(20, 8), dim=-1)
    centers = _kmeans(tokens, 4, generator)
    if tuple(centers.shape) != (4, 8):
        raise SystemExit(f"unexpected k-means shape {tuple(centers.shape)}")
    score = _sinkhorn_ot(centers, centers)
    if not torch.isfinite(score):
        raise SystemExit("Sinkhorn-OT returned a non-finite value")
    print(f"ot: k-means {tuple(centers.shape)}, self-OT {float(score):.4f}")


def check_gpu() -> None:
    import torch
    from PIL import Image

    from croa import config as C
    from croa.attacks import CLIPFamilyAttacks
    from croa.models import FeatureBank

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    bank = FeatureBank("cuda")
    family = CLIPFamilyAttacks(bank, "cuda")
    source = Image.open(C.BIGSCALE_DIR / "0.png").convert("RGB")
    target = Image.open(C.TARGET_DIR / "0.jpg").convert("RGB")
    result = family.attack_mattack_official(source, target, steps=2, seed=C.SEEDS[0])
    print(f"gpu: mattack_official steps=2 linf={result['linf']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="CROA-Attack repository smoke check")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="run two steps of M-Attack on pair 0 (needs CUDA and surrogate weights)",
    )
    args = parser.parse_args()
    check_layout()
    check_ot()
    if args.gpu:
        check_gpu()
    print("SMOKE-OK")


if __name__ == "__main__":
    main()
