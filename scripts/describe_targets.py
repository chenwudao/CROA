"""Official FOA protocol step 1: each victim describes the 100 TARGET images.

Output: outputs/descriptions/target_{victim}.jsonl  (resume-safe)
These victim-generated target descriptions are the REFERENCE for GPTScore,
exactly as in the official FOA-Attack gpt_evaluate.py (target_{model}.txt).
"""
import json
import sys
import time
from pathlib import Path

from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from croa import config as C
from croa.models import make_victim

DESC_DIR = C.OUT / "descriptions"
DESC_DIR.mkdir(parents=True, exist_ok=True)

ALL_VICTIMS = ["llava15-7b", "qwen2.5vl7b", "internvl3-8b",
               "gpt-5", "gemini-3.5-flash-lite", "claude-4.5-haiku"]


def main(victims=None, num_pairs=C.NUM_PAIRS):
    import torch
    victims = victims or ALL_VICTIMS
    pairs = C.load_pairs(num_pairs)
    for v_name in victims:
        out_fp = DESC_DIR / f"target_{v_name}.jsonl"
        done = set()
        if out_fp.exists():
            for line in out_fp.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try:
                        done.add(str(json.loads(line)["sample_id"]))
                    except Exception:
                        pass
        todo = [p for p in pairs if p["sample_id"] not in done]
        print(f"\n### Target-describe victim: {v_name} ({len(todo)} to go)", flush=True)
        if not todo:
            continue
        victim = make_victim(v_name, "cuda" if torch.cuda.is_available() else "cpu")
        fp = out_fp.open("a", encoding="utf-8")
        try:
            for p in todo:
                sid = p["sample_id"]
                tgt_pil = Image.open(p["target"]).convert("RGB")
                for attempt in range(4):
                    try:
                        cap = victim.ask(tgt_pil, C.CAPTION_PROMPT, max_new_tokens=32) or "none"
                        break
                    except Exception as e:
                        print(f"  retry {attempt} for {sid}: {type(e).__name__}", flush=True)
                        time.sleep(5 + 10 * attempt)
                else:
                    cap = "none"
                fp.write(json.dumps({"victim": v_name, "sample_id": sid,
                                     "caption": cap}) + "\n")
                fp.flush()
        finally:
            fp.close()
            del victim
            try:
                import torch as _t
                _t.cuda.empty_cache()
            except Exception:
                pass
    print("\n=== all target descriptions done ===", flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--victims", nargs="+", default=None)
    ap.add_argument("--num-pairs", type=int, default=C.NUM_PAIRS)
    a = ap.parse_args()
    main(a.victims, a.num_pairs)
