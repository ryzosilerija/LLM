"""Character n-gram baselines (bits per character) on val and test.

  python scripts/baselines.py                # data/tinystories
  python scripts/baselines.py --data data/mycorpus --max-order 8
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import data as D  # noqa: E402
from nllm import ngram  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/tinystories")
    ap.add_argument("--max-order", type=int, default=8)
    args = ap.parse_args()
    d = D.load(args.data)
    t0 = time.time()
    bl = ngram.baselines(d, args.max_order)
    splits = [s for s in ("val", "test") if s in bl]
    print(f"{bl['method']} n-gram models on {args.data} ({time.time() - t0:.1f} s)\n")
    print("order  " + "  ".join(f"{s + ' BPC':>9}" for s in splits))
    for k in sorted(bl[splits[0]], key=int):
        print(f"{k:>5}  " + "  ".join(f"{bl[s][k]:>9.3f}" for s in splits))
    for s in splits:
        o, v = ngram.best(bl, s)
        print(f"\nbest on {s}: {o}-gram, {v:.3f} BPC")


if __name__ == "__main__":
    main()
