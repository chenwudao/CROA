"""Generate perturbation visualizations from existing adversarial PNGs.

For chosen pairs and arms, computes delta = adv - source and renders a
5-panel row: source | adversarial | |delta| x10 amplified | delta heatmap | delta histogram stats.
Output: outputs/vis/{pair}_{arm}.png  (no GPU needed; pure numpy/PIL)
"""
import numpy as np
from PIL import Image
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ADV = ROOT / "outputs" / "adv"
DATA = ROOT / "data"
OUT = ROOT / "outputs" / "vis"
OUT.mkdir(parents=True, exist_ok=True)

ARMS = ["foa_official", "mattack_v2", "croa_tuned", "foa_v2_naive", "foa_v2"]
PAIRS = ["0", "7", "87", "51", "23", "66"]
AMPLIFY = 10

FONT = None
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

def load(name):
    return np.asarray(Image.open(name).convert("RGB"), dtype=np.float32)

def render(sid, arm):
    src = load(DATA / "bigscale" / f"{sid}.png")
    adv_path = ADV / arm / f"{sid}.png"
    if not adv_path.exists():
        return False
    adv = load(adv_path)
    if adv.shape != src.shape:
        adv = np.asarray(Image.fromarray(adv.astype(np.uint8)).resize(
            (src.shape[1], src.shape[0])), dtype=np.float32)
    delta = adv - src                                   # [-255, 255]
    amp = np.clip(128 + delta * AMPLIFY, 0, 255).astype(np.uint8)
    linf = np.abs(delta).max() / 255.0
    rms = np.sqrt((delta ** 2).mean()) / 255.0

    if HAVE_MPL:
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.6))
        axes[0].imshow(src.astype(np.uint8)); axes[0].set_title("source", fontsize=11)
        axes[1].imshow(adv.astype(np.uint8)); axes[1].set_title(f"{arm} adv", fontsize=11)
        axes[2].imshow(amp); axes[2].set_title(f"|delta| x{AMPLIFY}", fontsize=11)
        hm = axes[3].imshow(np.abs(delta).mean(axis=2), cmap="inferno")
        axes[3].set_title(f"delta heat  (Linf={linf:.3f}, RMS={rms:.3f})", fontsize=11)
        plt.colorbar(hm, ax=axes[3], fraction=0.046)
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(OUT / f"{sid}_{arm}.png", dpi=150)
        plt.close(fig)
    else:
        row = Image.new("RGB", (4 * 224 + 50, 250), "white")
        for i, im in enumerate([src, adv, amp]):
            row.paste(Image.fromarray(im.astype(np.uint8)).resize((224, 224)), (i * 234 + 8, 12))
        row.save(OUT / f"{sid}_{arm}.png")
    return True

def main():
    n = 0
    for sid in PAIRS:
        for arm in ARMS:
            if render(sid, arm):
                n += 1
                print(f"ok {sid}_{arm}", flush=True)
    print(f"done: {n} visualizations -> {OUT}")

if __name__ == "__main__":
    main()
