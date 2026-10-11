"""Length generalization: validation loss at contexts longer than training.

  python scripts/eval_length.py --ids 01,17,18 --lengths 128,256,512

Learned absolute positions cannot represent unseen positions, so they are
reported as n/a beyond the training context. Sinusoidal and RoPE can be
evaluated at any length; how much their loss degrades is the result.
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import data as D  # noqa: E402
from nllm.train import evaluate, load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--ids", default="01,17,18")
    ap.add_argument("--lengths", default="128,256,512")
    ap.add_argument("--batches", type=int, default=10)
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    results = {}
    for cid in args.ids.split(","):
        for d in sorted(glob.glob(os.path.join(args.runs, f"{cid}_*"))):
            if not os.path.exists(os.path.join(d, "model.npz")):
                continue
            model, cfg, _ = load_model(d)
            data = D.load(cfg["data_dir"])
            row = {}
            for L in lengths:
                if cfg["positional"] == "learned" and L > cfg["ctx"]:
                    row[L] = None
                    continue
                row[L] = evaluate(model, data, {**cfg, "batch_size": max(1, cfg["batch_size"] * cfg["ctx"] // L)},
                                  n_batches=args.batches, ctx=L)["loss"]
            results[os.path.basename(d)] = row
            cells = "  ".join("   n/a  " if v is None else f"{v:.4f}" for v in row.values())
            print(f"{os.path.basename(d):<48} {cells}")
    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/length_generalization.json", "w"), indent=2)
    print("\nwrote results/length_generalization.json   (columns: " + ", ".join(map(str, lengths)) + ")")


if __name__ == "__main__":
    main()
