"""Blind human rating of generated text.

  python scripts/rate_samples.py results/runs/01_..._s1 results/runs/21_..._s1 [more runs]

Generates samples from each model with the same prompts and seeds, shuffles
them so you don't know which model wrote what, and asks for two scores:

  fluency    1 = gibberish ... 5 = reads like real English sentences
  coherence  1 = no story at all ... 5 = a sensible, connected story

Answers are appended to results/human_ratings.json (you can stop with q and
continue later; ask friends to rate too with --rater NAME). The dashboard and
final report show the averages per run.
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm.metrics import DEFAULT_PROMPTS, generate  # noqa: E402
from nllm.train import load_model  # noqa: E402


def ask(prompt):
    while True:
        a = input(prompt).strip().lower()
        if a in ("q", "s"):
            return a
        if a in ("1", "2", "3", "4", "5"):
            return int(a)
        print("  enter 1-5, s to skip, q to quit")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--per-run", type=int, default=5, help="samples per run")
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--out", default="results/human_ratings.json")
    ap.add_argument("--rater", default="me")
    args = ap.parse_args()

    samples = []
    for run_dir in args.runs:
        model, cfg, chars = load_model(run_dir)
        stoi = {c: i for i, c in enumerate(chars)}
        prompts = cfg.get("gen_prompts") or DEFAULT_PROMPTS
        for k in range(args.per_run):
            p = prompts[k % len(prompts)]
            ids, _ = generate(model, cfg["ctx"], [stoi[c] for c in p if c in stoi], args.chars,
                              seed=1000 + k)  # different seeds from the report samples
            samples.append({"run": os.path.basename(os.path.normpath(run_dir)), "prompt": p,
                            "text": "".join(chars[i] for i in ids)})
        print(f"generated {args.per_run} samples from {run_dir}")
    random.Random(time.time()).shuffle(samples)

    ratings = json.load(open(args.out)) if os.path.exists(args.out) else []
    print(f"\n{len(samples)} samples, shown in random order. 1-5 per question, s = skip, q = quit and save.\n")
    try:
        for i, smp in enumerate(samples, 1):
            print("=" * 72 + f"\n[{i}/{len(samples)}]\n\n{smp['prompt']}{smp['text']}\n")
            f = ask("fluency 1-5: ")
            if f == "q":
                break
            if f == "s":
                continue
            c = ask("coherence 1-5: ")
            if c == "q":
                break
            if c == "s":
                continue
            ratings.append({**smp, "fluency": f, "coherence": c, "rater": args.rater,
                            "time": time.strftime("%Y-%m-%d %H:%M")})
    finally:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(ratings, open(args.out, "w"), indent=2)
        print(f"\nsaved {len(ratings)} ratings to {args.out}")
    by = {}
    for r in ratings:
        by.setdefault(r["run"], []).append(r)
    for run, rs in sorted(by.items()):
        print(f"  {run:<48} n={len(rs):<3} fluency {sum(r['fluency'] for r in rs) / len(rs):.2f}  "
              f"coherence {sum(r['coherence'] for r in rs) / len(rs):.2f}")


if __name__ == "__main__":
    main()
