"""Generate result visualizations for the CROA ICASSP paper.

fig4_commercial_bars.png : grouped bars, commercial victims x 6 methods (ASR)
fig5_radar.png           : radar over 6 victims, 5 methods (ASR)
fig6_ablation_bars.png   : component ablation bars (commercial victims)
Data source: outputs/paper_audit_20260922/summary.csv (consistently ordered
model-judge scoring, threshold 0.5).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5,
    "axes.linewidth": 0.6, "xtick.direction": "out",
})

OUT = Path(__file__).resolve().parent / "assets"
OUT.mkdir(exist_ok=True)

COLORS = {  # colorblind-safe, distinct
    "attackvlm":       "#9e9e9e",
    "anyattack":       "#8d6e63",
    "mattack":         "#1f77b4",
    "foa":             "#2ca02c",
    "mattackv2":       "#9467bd",
    "croa":            "#d62728",
}
HATCH = {"attackvlm": "//", "anyattack": "xx", "mattack": "",
         "foa": "//", "mattackv2": "xx", "croa": ""}
LABEL = {"attackvlm": "AttackVLM", "anyattack": "AnyAttack",
         "mattack": "M-Attack", "foa": "FOA-Attack",
         "mattackv2": "M-Attack-V2 (restr.)", "croa": "CROA (ours)"}

# Load the same audited scores as the manuscript tables; no hand-entered ASR.
import csv
with (OUT.parent.parent / "outputs/paper_audit_20260922/summary.csv").open() as f:
    _rows = list(csv.DictReader(f))
_scores = {(r["victim"], r["arm"]): float(r["asr"]) for r in _rows if r["subset"] == "all100"}
_arm_ids = {"attackvlm":"attackvlm", "anyattack":"anyattack", "mattack":"mattack_official",
            "foa":"foa_official", "mattackv2":"mattack_v2", "croa":"croa_tuned"}
COMMERCIAL = {label: {a: _scores[(v, arm)] for a, arm in _arm_ids.items()}
              for label, v in [("GPT-5","gpt-5"),("Gemini-3.5-FL","gemini-3.5-flash-lite"),("Claude-4.5-H","claude-4.5-haiku")]}
OPENSOURCE = {label: {a: _scores[(v, arm)] for a, arm in _arm_ids.items()}
              for label, v in [("LLaVA-1.5","llava15-7b"),("Qwen2.5-VL","qwen2.5vl7b"),("InternVL3","internvl3-8b")]}
ABLATION = {label: tuple(_scores[(v, arm)] for v in ["gpt-5","gemini-3.5-flash-lite","claude-4.5-haiku"])
            for label, arm in [("CLIP+MCA","foa_mca_4crop"),("w/o DINO","croa_no_dino"),
                               ("w/o local OT","croa_no_local"),("w/o repulsion","croa_no_rep"),("Full CROA","croa_tuned")]}
VICTIMS = ["GPT-5", "Gemini-3.5-FL", "Claude-4.5-H"]
ARMS = ["attackvlm", "anyattack", "mattack", "foa", "mattackv2", "croa"]

# ---------------- Combined: bars (left two panels) + radar (right) ----------------
fig = plt.figure(figsize=(7.05, 2.75))
gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1], wspace=0.16)

gs_bars = gs[0].subgridspec(1, 2, wspace=0.06)
for j, (data, title) in enumerate([(COMMERCIAL, "Commercial victims"),
                                   (OPENSOURCE, "Open-source victims")]):
    ax = fig.add_subplot(gs_bars[j])
    victims = list(data.keys())
    x = np.arange(len(victims)); w = 0.13
    for i, arm in enumerate(ARMS):
        vals = [data[v][arm] for v in victims]
        ax.bar(x + (i - 2.5) * w, vals, w * 0.92, color=COLORS[arm],
               hatch=HATCH[arm], edgecolor="black", linewidth=0.4,
               label=LABEL[arm] if j == 0 else None)
        for xi, v in zip(x + (i - 2.5) * w, vals):
            ax.text(xi, v + 1.2, str(v), ha="center", fontsize=4.8)
    ax.set_xticks(x)
    ax.set_xticklabels(victims, fontsize=6.4, rotation=22, ha="right")
    ax.set_ylim(0, 104)
    ax.set_title(title, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linewidth=0.35, alpha=0.4); ax.set_axisbelow(True)
    if j == 0:
        ax.set_ylabel("Targeted ASR (%)", fontsize=8)
        ax.set_yticks([0, 20, 40, 60, 80, 100])
    else:
        ax.set_yticks([])

axr = fig.add_subplot(gs[1], polar=True)
cats = ["GPT-5", "Gemini-3.5-FL", "Claude-4.5-H", "LLaVA-1.5", "Qwen2.5-VL", "InternVL3"]
N = len(cats); angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist(); angles += angles[:1]
for arm in ["mattack", "foa", "mattackv2", "croa"]:
    vals = [COMMERCIAL[c][arm] for c in COMMERCIAL] + [OPENSOURCE[o][arm] for o in ["LLaVA-1.5", "Qwen2.5-VL", "InternVL3"]]
    vals += vals[:1]
    axr.plot(angles, vals, linewidth=1.2, color=COLORS[arm],
             linestyle="--" if arm != "croa" else "-", label=LABEL[arm])
    axr.fill(angles, vals, color=COLORS[arm], alpha=0.06)
axr.set_xticks(angles[:-1])
axr.set_xticklabels(cats, fontsize=5.6)
axr.tick_params(axis="x", pad=-4)
axr.set_ylim(0, 100); axr.set_yticks([20, 40, 60, 80])
axr.set_yticklabels(["20", "40", "60", "80"], fontsize=5.2, color="#777777")
axr.grid(linewidth=0.4, alpha=0.5); axr.spines["polar"].set_linewidth(0.5)
axr.set_title("ASR across all victims", fontsize=8.5, pad=10)

fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=COLORS[a], hatch=HATCH[a], ec="black", lw=0.4)
                    for a in ARMS],
           labels=[LABEL[a] for a in ARMS], ncol=6, loc="upper center",
           bbox_to_anchor=(0.42, 1.03), fontsize=6.8, frameon=False,
           handlelength=1.3, columnspacing=1.0)
fig.subplots_adjust(left=0.085, right=0.965, top=0.885, bottom=0.17, wspace=0.14)
fig.savefig(OUT / "fig4_results_combined.png", dpi=300)
plt.close(fig)
print("combined figure saved")
