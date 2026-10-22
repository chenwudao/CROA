"""Generate Fig. 1 bottom: Single-column MPCAttack-style radar with unified Times serif font and refined line weights."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parent

# 1. Prepare data
vs = ['gpt-5', 'gemini-3.5-flash-lite', 'claude-4.5-haiku', 'llava15-7b', 'qwen2.5vl7b', 'internvl3-8b']
v_labels = ['GPT-5', 'Gemini-3.5-\nFlash-Lite', 'Claude-4.5-\nHaiku', 'LLaVA-1.5-7B', 'Qwen2.5-VL-7B', 'InternVL3-8B']

# Exact audited numbers
data = {
    'croa_tuned': [90.0, 56.0, 7.0, 95.0, 85.0, 96.0],
    'mpc_official': [89.0, 49.0, 5.0, 85.0, 75.0, 95.0],
    'foa_official': [66.0, 30.0, 4.0, 73.0, 67.0, 86.0],
    'mattack_official': [52.0, 19.0, 1.0, 79.0, 65.0, 72.0],
    'anyattack': [11.0, 3.0, 3.0, 14.0, 12.0, 13.0],
    'attackvlm': [0.0, 0.0, 0.0, 3.0, 0.0, 1.0],
}

# Refined thinner lines and markers, MPCAttack aesthetic
methods = [
    ('croa_tuned', 'CROA (ours)', '#c62828', '-', 1.4, 'o', 2.5),
    ('mpc_official', 'MPC-A', '#4527a0', '-.', 1.05, 's', 2.2),
    ('foa_official', 'FOA', '#2e7d32', '--', 0.95, '^', 2.2),
    ('mattack_official', 'M-Attack', '#1565c0', '--', 0.95, 'd', 2.2),
    ('anyattack', 'AnyAttack', '#ef6c00', ':', 0.85, 'v', 1.9),
    ('attackvlm', 'AttackVLM', '#546e7a', ':', 0.80, 'x', 1.9),
]

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Nimbus Roman', 'Liberation Serif', 'DejaVu Serif', 'Times New Roman'],
    'font.size': 7.5,
    'pdf.fonttype': 42
})

W, H = 3.35, 1.82
fig = plt.figure(figsize=(W, H), facecolor='white', dpi=300)

ax_radar = fig.add_axes([0.02, 0.07, 0.51, 0.86], projection='polar')
ax_radar.set_theta_offset(np.pi / 2)
ax_radar.set_theta_direction(-1)
theta = np.arange(len(vs)) * 2 * np.pi / len(vs)

ax_radar.set_xticks(theta)
ax_radar.set_xticklabels(v_labels, fontsize=6.2, weight='bold', color='#1a1a1a')
ax_radar.tick_params(axis='x', pad=3.2)
ax_radar.set_ylim(0, 100)
ax_radar.set_yticks([25, 50, 75, 100])
ax_radar.set_yticklabels(['25', '50', '75', '100'], fontsize=5.0, color='#666666')
ax_radar.set_rlabel_position(30)
ax_radar.grid(color='#d8dde3', lw=0.45, linestyle='-')
ax_radar.spines['polar'].set_color('#8c959f')
ax_radar.spines['polar'].set_linewidth(0.55)

handles = []
for arm, label, color, style, lw, marker, ms in methods:
    values = data[arm]
    mean_val = np.mean(values)
    line, = ax_radar.plot(
        np.r_[theta, theta[0]],
        values + values[:1],
        color=color,
        ls=style,
        lw=lw,
        marker=marker,
        markersize=ms,
        label=f'{label:<10} {mean_val:.1f}%'
    )
    if arm == 'croa_tuned':
        ax_radar.fill(np.r_[theta, theta[0]], values + values[:1], color=color, alpha=0.08)
    handles.append(line)

fig.legend(
    handles=handles,
    loc='center left',
    bbox_to_anchor=(0.58, 0.50),
    frameon=True,
    facecolor='#f8f9fa',
    edgecolor='#d0d7de',
    framealpha=0.96,
    fontsize=5.8,
    handlelength=1.5,
    handletextpad=0.35,
    labelspacing=0.36,
    title='Mean ASR',
    title_fontsize=6.4,
    ncol=1
)

out_png = root / 'assets/fig1_radar_enlarged.png'
fig.savefig(out_png, dpi=300, bbox_inches='tight', pad_inches=0.02)
print(f'Successfully regenerated: {out_png}')
