"""Re-score saved captions consistently; never alter legacy response/score files.

Run on the experiment host. API calls are only made with --rescore; otherwise
missing cache entries fail closed. Outputs include 100-pair and non-pilot
90-pair summaries. The latter are sensitivity analyses, not a new blind test.
"""
import argparse
import collections
import concurrent.futures
import csv
import hashlib
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from croa import config as C
VICTIMS = ['gpt-5', 'gemini-3.5-flash-lite', 'claude-4.5-haiku',
           'llava15-7b', 'qwen2.5vl7b', 'internvl3-8b']
ARMS = ['attackvlm', 'anyattack', 'mattack_official', 'foa_official',
        'mattack_v2', 'croa_tuned', 'mpc_official', 'croa_no_dino',
        'croa_no_local', 'croa_no_rep', 'foa_mca_4crop']


def jsonl(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]


def write_csv(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rescore', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--include-expanded-v2', action='store_true',
                        help='Include the expanded M-Attack-V2 adaptation only after all four victims have 100 responses.')
    args = parser.parse_args()
    selected_arms = ARMS + (['mattack_v2_full'] if args.include_expanded_v2 else [])
    out = ROOT / 'outputs/paper_audit_20260922'
    out.mkdir(parents=True, exist_ok=True)
    cache = {}
    for path in [ROOT/'outputs/gptscore_cache.jsonl', out/'gptscore_cache.jsonl']:
        cache.update({r['k']: r['s'] for r in jsonl(path)})
    kmr_cache = {r['k']: r['m'] for r in jsonl(ROOT/'outputs/gptkmr_cache.jsonl')}
    keywords = C.load_keywords()
    scorer = None
    if args.rescore:
        import scripts.run_benchmark as rb
        rb.GPT_CACHE_PATH = out/'gptscore_cache.jsonl'
        scorer = rb.GPTScorer()
        scorer.cache.update(cache)
    rows, issues = [], []
    for victim in VICTIMS:
        refs = {str(r['sample_id']): r['caption'] for r in
                jsonl(ROOT/f'outputs/descriptions/target_{victim}.jsonl')}
        records = {}
        for r in jsonl(ROOT/f'outputs/responses/responses_{victim}.jsonl'):
            if r['arm'] in selected_arms:
                key = (r['arm'], str(r['sample_id']))
                if key in records and records[key]['caption'] != r['caption']:
                    issues.append({'victim': victim, 'arm': key[0], 'sample_id': key[1],
                                   'issue': 'different duplicate captions; latest used'})
                records[key] = r
        expected_ids = {str(i) for i in range(100)}
        for arm in selected_arms:
            ids = {sid for a, sid in records if a == arm}
            required_v2 = (args.include_expanded_v2 and arm == 'mattack_v2_full'
                           and victim in VICTIMS[:4])
            if (ids or required_v2) and ids != expected_ids:
                raise ValueError(f'Incomplete/invalid sample set: {victim}/{arm}: '
                                 f'{len(ids)} samples. No partial benchmark summary is valid.')
        old = {}
        fp = ROOT/f'outputs/responses/scored_{victim}.csv'
        if fp.exists():
            with fp.open(encoding='utf-8-sig', newline='') as f:
                old = {(r['arm'], str(r['sample_id'])): r for r in csv.DictReader(f)}

        def score(item):
            (arm, sid), r = item
            cap, ref = r['caption'], refs.get(sid, '')
            if not ref.strip():
                raise ValueError(f'Missing caption/reference: {victim}/{arm}/{sid}')
            k = hashlib.sha256(f'{C.GPTSCORE_MODEL}|{cap}|{ref}'.encode()).hexdigest()
            empty = not cap.strip() or cap.strip().lower() == 'none'
            if empty:
                sim = 0.0  # Retain failed/empty generations in the denominator.
            elif k in cache:
                sim = float(cache[k])
            elif scorer:
                sim = scorer.score(cap, ref)
            else:
                raise ValueError(f'Uncached score: {victim}/{arm}/{sid}; use --rescore')
            previous = old.get((arm, sid), {})
            kws = keywords.get(sid, [])
            kw_str = json.dumps(sorted(kws), ensure_ascii=False)
            kw_key = hashlib.sha256(f'gpt-4o-mini|{kw_str}|{cap}'.encode()).hexdigest()
            kmr = ''
            if kw_key in kmr_cache and kws:
                value = len(kmr_cache[kw_key]) / len(kws)
                prior = previous.get('kmr_gpt', '') if previous.get('caption') == cap else ''
                kmr = prior if prior != '' and abs(float(prior)-value) < 0.0001 else value
            elif empty:
                kmr = 0.0
            return dict(victim=victim, arm=arm, sample_id=int(sid), score=sim,
                        is_success=int(sim >= C.GPTSCORE_THRESHOLD), kmr_gpt=kmr,
                        direction='caption_then_victim_target', status='empty_generation' if empty else 'scored', caption=cap)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            scored = list(pool.map(score, records.items()))
        rows.extend(scored)
        write_csv(out/f'scored_{victim}.csv', scored)
        print(f'Completed {victim}: {len(scored)} scores', flush=True)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[(r['victim'], r['arm'])].append(r)
    summary = []
    for (v, a), rs in groups.items():
        for subset, subset_rows in [('all100',rs),('exclude_pilot10',[r for r in rs if r['sample_id']>=10])]:
            n = len(subset_rows)
            if not n:
                continue
            kmr = [float(r['kmr_gpt']) for r in subset_rows if r['kmr_gpt']!='']
            summary.append(dict(victim=v,arm=a,subset=subset,n=n,
                asr=100*sum(r['is_success'] for r in subset_rows)/n,
                kmr=100*sum(kmr)/len(kmr) if len(kmr)==n else '',kmr_n=len(kmr)))
    write_csv(out/'summary.csv',summary)
    paired=[]
    for v in VICTIMS:
        full={r['sample_id']:r['is_success'] for r in groups[(v,'croa_tuned')]}
        for arm in ['mpc_official','croa_no_dino','croa_no_local','croa_no_rep','foa_mca_4crop']:
            base={r['sample_id']:r['is_success'] for r in groups[(v,arm)]}
            for subset,ids in [('all100',set(full)&set(base)),('exclude_pilot10',{i for i in set(full)&set(base) if i>=10})]:
                if not ids:continue
                w=sum(full[i]>base[i] for i in ids);l=sum(full[i]<base[i] for i in ids)
                p=min(1,2*sum(math.comb(w+l,k) for k in range(min(w,l)+1))/2**(w+l)) if w+l else 1
                paired.append(dict(victim=v,baseline=arm,subset=subset,n=len(ids),wins=w,losses=l,p_unadjusted=p))
    write_csv(out/'paired.csv',paired)
    (out/'issues.json').write_text(json.dumps(issues,indent=2),encoding='utf-8')
    print('DONE',out,flush=True)


if __name__ == '__main__':
    main()
