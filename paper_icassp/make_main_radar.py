"""Compact Fig.1 radar: raw ASR from the audited common-victim table."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
rows=list(csv.DictReader((root.parent/'outputs/paper_audit_20260922/summary.csv').open()))
data={(r['victim'],r['arm']):r for r in rows if r['subset']=='all100'}
vs=['gpt-5','gemini-3.5-flash-lite','claude-4.5-haiku','llava15-7b','qwen2.5vl7b','internvl3-8b']
methods=[
 ('attackvlm','AttackVLM','#b0b0b0',':'),
 ('anyattack','AnyAttack','#a1887f',':'),
 ('mattack_official','M-Attack','#4c72b0','--'),
 ('foa_official','FOA','#55a868','--'),
 ('mpc_official','MPC-A','#cc8964','-.'),
 ('croa_tuned','CROA (ours)','#c9364f','-'),
]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':7,'pdf.fonttype':42})
fig=plt.figure(figsize=(3.4,2.4),facecolor='white')
ax=fig.add_axes([.10,.12,.48,.74],projection='polar')
ax.set_theta_offset(np.pi/2);ax.set_theta_direction(-1)
theta=np.arange(len(vs))*2*np.pi/len(vs)
ax.set_xticks(theta);ax.set_xticklabels(['GPT-5','Gemini','Claude','LLaVA','Qwen','InternVL'],fontsize=6.5)
ax.tick_params(axis='x',pad=0)
ax.set_ylim(0,100);ax.set_yticks([25,50,75,100]);ax.set_yticklabels(['25','50','75','100'],fontsize=5.2,color='#727780');ax.set_rlabel_position(40)
ax.grid(color='#d9dde2',lw=.55);ax.spines['polar'].set_color('#bec5cd');ax.spines['polar'].set_linewidth(.6)
handles=[]
for arm,label,color,style in methods:
 rs=[data.get((v,arm)) for v in vs]
 if None in rs:
  print(f'skip {arm}: partial victims'); continue
 assert all(int(r['n'])==100 for r in rs)
 values=[float(r['asr']) for r in rs];line,=ax.plot(np.r_[theta,theta[0]],values+values[:1],color=color,ls=style,lw=1.5 if arm=='croa_tuned' else 1.05,marker='o',markersize=2.2,label=f'{label}  {np.mean(values):.2f}')
 if arm=='croa_tuned':ax.fill(np.r_[theta,theta[0]],values+values[:1],color=color,alpha=.07)
 handles.append(line)
fig.legend(handles=handles,loc='center left',bbox_to_anchor=(.62,.52),frameon=False,fontsize=5.8,handlelength=1.6,handletextpad=.4,labelspacing=.55,title='Mean ASR (%)',title_fontsize=6.2,ncol=1)
fig.savefig(root/'assets/fig1_main_radar.pdf')
fig.savefig(root/'assets/fig1_main_radar.png',dpi=300)
