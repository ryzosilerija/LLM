"""Learning-rate sweep: short runs over a log-spaced grid, pick the best.

  # sweep every optimizer on the baseline architecture, save winners:
  python scripts/lr_sweep.py --write

  # sweep one optimizer with a custom grid
  python scripts/lr_sweep.py --optimizers muon --grid 1e-3,3e-3,1e-2,3e-2

  # sweep a single config (e.g. a no-norm run) and store a per-config LR
  python scripts/lr_sweep.py --config-id 15 --write

By default the grid is the current LR in lrs.json times 1/10, 1/3, 1, 3.
If the best LR lands on the edge of the grid, the script warns you:
extend the grid in that direction and sweep again.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import experiments as E  # noqa: E402
from nllm.optim import OPTIMIZERS  # noqa: E402
from nllm.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="experiments/configs.json")
    ap.add_argument("--lrs", default="experiments/lrs.json")
    ap.add_argument("--out", default="results/sweeps")
    ap.add_argument("--optimizers", default=",".join(OPTIMIZERS))
    ap.add_argument("--config-id", default=None, help="sweep this config instead of the baseline")
    ap.add_argument("--grid", default=None, help="comma-separated LRs")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--write", action="store_true", help="save best LRs into lrs.json")
    args = ap.parse_args()

    exp = E.load(args.configs)
    lrs_all = json.load(open(args.lrs))
    lrs = E.load_lrs(args.lrs)
    by_id = {c["id"]: c for c in exp["configs"]}

    if args.config_id:
        targets = [(f"config:{args.config_id}", by_id[args.config_id])]
    else:
        base = by_id["01"]
        targets = [(o, {**base, "optimizer": o}) for o in args.optimizers.split(",")]

    overrides = {"max_steps": args.steps, "warmup": max(10, args.steps // 20),
                 "eval_every": args.steps, "log_every": args.steps, "save_model": False,
                 "final_eval": False, "write_report": False, "train_eval": False,
                 **E.parse_overrides(args.set)}

    for key, entry in targets:
        center = lrs.get(key, lrs[entry["optimizer"]])
        grid = ([float(x) for x in args.grid.split(",")] if args.grid
                else [center / 10, center / 3, center, center * 3])
        print(f"\n== {key} ({entry['optimizer']}), grid {[f'{g:.1e}' for g in grid]}")
        results = []
        for lr in grid:
            cfg = E.resolve(exp, entry, lrs, {**overrides, "lr": lr, "seed": 1})
            out = os.path.join(args.out, f"{key.replace(':', '_')}_lr{lr:.1e}")
            if E.is_done(out):
                s = json.load(open(os.path.join(out, "summary.json")))
            else:
                s = train(cfg, out, verbose=False)
            v = s["final_val_loss"]
            results.append((lr, v))
            print(f"   lr {lr:.1e}  ->  {'diverged' if s['diverged'] else f'val {v:.4f}'}")
        ok = [(lr, v) for lr, v in results if math.isfinite(v)]
        if not ok:
            print("   every LR diverged; try a lower grid")
            continue
        best_lr, best_v = min(ok, key=lambda t: t[1])
        print(f"   best lr {best_lr:.1e} (val {best_v:.4f})")
        if best_lr in (grid[0], grid[-1]):
            print("   WARNING: best LR is at the edge of the grid; extend it and sweep again")
        if args.write:
            lrs_all[key] = float(f"{best_lr:.3g}")

    if args.write:
        json.dump(lrs_all, open(args.lrs, "w"), indent=2)
        print(f"\nwrote {args.lrs}")


if __name__ == "__main__":
    main()
