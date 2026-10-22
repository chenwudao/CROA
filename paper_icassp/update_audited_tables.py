"""Generate final-paper tables from verified 100-pair scores; no sensitivity table."""
import csv,re
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT.parent/'outputs/paper_audit_20260922/summary.csv').open()))
data={(r['victim'],r['arm']):r for r in rows if r['subset']=='all100'}
vs=['gpt-5','gemini-3.5-flash-lite','claude-4.5-haiku','llava15-7b']
methods=[('attackvlm',r'AttackVLM~\cite{attackvlm}'),('anyattack',r'AnyAttack~\cite{anyattack}'),('mattack_official',r'M-Attack~\cite{mattack}'),('foa_official',r'FOA~\cite{foa}'),('mattack_v2',r'V2-R~\cite{mattackv2}'),('mattack_v2_full',r'V2-E~\cite{mattackv2}'),('mpc_official',r'MPC-A~\cite{mpcattack}'),('croa_tuned',r'\ours{}')]
def val(v,a,m):
 r=data.get((v,a))
 if r is None:return None
 assert int(r['n'])==100,(v,a,r['n'])
 assert r[m]!='',(v,a,m)
 return float(r[m])
def row(label,vals):return label+' & '+' & '.join(vals)+r' \\'+'\n'
def mean(a,m,victims):
 xs=[val(v,a,m) for v in victims]
 return sum(xs)/len(xs) if all(x is not None for x in xs) else None
def fmt(x,precision=1,bold=False):
 if x is None:return '--'
 text=f'{x:.{precision}f}'
 return r'\textbf{'+text+'}' if bold else text
main=r'''\begin{table*}[t]
\centering
\caption{Main comparison on four common victims: ASR / GPT-KMR (\%), 100 pairs per victim. Mean gives equal weight to each victim. V2-R: restricted M-Attack-V2; V2-E: expanded adaptation with ATA/PE+; MPC-A: adapted MPCAttack. Implementation differences are specified in Sec.~3.1. Dashes denote unavailable verified KMR; KMR means require four scores. Bold marks column maxima.}
\label{tab:main}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lrrrrrrrrrr}
\toprule
& \multicolumn{2}{c}{GPT-5} & \multicolumn{2}{c}{Gemini-3.5-FL} & \multicolumn{2}{c}{Claude-4.5-H} & \multicolumn{2}{c}{LLaVA-1.5} & \multicolumn{2}{c}{Mean} \\
Method & ASR & KMR & ASR & KMR & ASR & KMR & ASR & KMR & ASR & KMR \\
\midrule
'''
# Older open-source KMR entries without verified caches remain unreported.
for a,label in methods:
 cells=[]
 for v in vs:
  for metric in ['asr','kmr']:
   r=data[(v,a)];x=float(r[metric]) if r[metric]!='' else None
   available=[float(data[(v,b)][metric]) for b,_ in methods if data[(v,b)][metric]!='']
   cells.append(fmt(x,1,x is not None and x==max(available)))
 for metric in ['asr','kmr']:
  def safe_mean(b):
   rs=[data[(v,b)][metric] for v in vs]
   return sum(float(x) for x in rs)/len(rs) if all(x!='' for x in rs) else None
  x=safe_mean(a);available=[safe_mean(b) for b,_ in methods if safe_mean(b) is not None]
  cells.append(fmt(x,2,x is not None and x==max(available)))
 main+=row(label,cells)
main+=r'\bottomrule'+'\n'+r'\end{tabular}'+'\n'+r'\end{table*}'
opens=r'''\begin{table}[t]
\centering
\caption{Open-source ASR (\%). Dashes denote unevaluated combinations; a three-model mean is reported only for complete rows.}
\label{tab:opensource}
\small
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lrrrr}
\toprule
Method & LLaVA & Qwen2.5 & InternVL3 & Mean \\
\midrule
'''
os=['llava15-7b','qwen2.5vl7b','internvl3-8b']
for a,label in methods:
 xs=[val(v,a,'asr') for v in os]
 opens+=row(label,[fmt(x,1,a=='croa_tuned') for x in xs]+[fmt(mean(a,'asr',os),1,a=='croa_tuned')])
opens+='\\bottomrule\n\\end{tabular}\n\\end{table}'
arms=[('foa_official','FOA (CLIP)'),('foa_mca_4crop','CLIP + MCA'),('croa_no_dino','CROA w/o DINOv2'),('croa_no_local','CROA w/o local OT'),('croa_no_rep','CROA w/o repulsion'),('croa_tuned',r'Full \ours{}')]
abl=r'''\begin{table}[t]
\centering
\caption{Component analysis: ASR (\%) on the same four victims. The first two rows compare configurations; the final four isolate component removals. Bold marks column maxima among these variants.}
\label{tab:ablation}
\small
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lrrrrr}
\toprule
Variant & GPT-5 & Gemini & Claude & LLaVA & Mean \\
\midrule
'''
for idx,(a,label) in enumerate(arms):
 if idx==2:abl+='\\midrule\n'
 xs=[val(v,a,'asr') for v in vs];av=mean(a,'asr',vs)
 cells=[fmt(x,0,x==max(val(v,b,'asr') for b,_ in arms)) for v,x in zip(vs,xs)]
 abl+=row(label,cells+[fmt(av,2,av==max(mean(b,'asr',vs) for b,_ in arms))])
abl+='\\bottomrule\n\\end{tabular}\n\\end{table}'
p=ROOT/'main.tex';s=p.read_text()
for name,table in [('MAIN',main),('OPEN',opens),('ABLATION',abl)]:
 marker='% AUDITED_'+name+'_TABLE';end='% END_AUDITED_'+name+'_TABLE'
 if end in s:s=re.sub(re.escape(marker)+r'.*?'+re.escape(end),lambda m:marker+'\n'+table+'\n'+end,s,flags=re.S)
 else:
  assert s.count(marker)==1
  s=s.replace(marker,marker+'\n'+table+'\n'+end)
p.write_text(s)
print('Generated common-victim, open-source, and ablation tables; no sensitivity analysis')
