"""Aggregate all finished runs.

  python scripts/analyze.py              # tables, charts, dashboard
  python scripts/analyze.py --reports    # also regenerate every run's report.html

Outputs
  results/dashboard.html    everything in one page (sortable leaderboard + charts)
  results/runs.csv          one row per run, every metric
  results/configs.csv       one row per config, mean/std over seeds, ranked
  results/charts/*.png      the dashboard charts as separate images (for a README or writeup)
It also prints the top configs and the command to give the top 5 extra seeds.
"""
import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import report as R  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reports", action="store_true", help="regenerate every run's report.html")
    args = ap.parse_args()

    runs = R.load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no finished runs in {args.runs}")
    configs = R.aggregate(runs)
    os.makedirs(args.out, exist_ok=True)
    keys = [m[0] for m in R.METRICS]

    with open(os.path.join(args.out, "runs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "id", "seed", "optimizer", "activation", "norm", "positional", "lr", "diverged"] + keys)
        for r in runs:
            c = r["cfg"]
            w.writerow([r["name"], r["id"], c["seed"], c["optimizer"], c["activation"], c["norm"], c["positional"],
                        c["lr"], r["s"].get("diverged")] + [R.run_value(r, k) for k in keys])
    with open(os.path.join(args.out, "configs.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rank", "id", "optimizer", "activation", "norm", "positional", "lr", "n_seeds", "n_diverged"] +
                   [x for k in keys for x in (k, k + "_std")])
        for i, c in enumerate(configs, 1):
            w.writerow([i, c["id"], c["optimizer"], c["activation"], c["norm"], c["positional"], c["lr"],
                        c["n_seeds"], c["n_diverged"]] + [c[x] for k in keys for x in (k, k + "_std")])

    dash = R.build_dashboard(args.runs, os.path.join(args.out, "dashboard.html"),
                             charts_dir=os.path.join(args.out, "charts"))
    if args.reports:
        for r in runs:
            R.run_report(r["dir"], all_runs=runs)
        print(f"regenerated {len(runs)} run reports")

    print(f"{len(runs)} runs, {len(configs)} configs\n")
    print(f"{'rank':>4}  {'config':<45} {'val loss':>17}  {'BPC':>6}  seeds")
    for k, c in enumerate(configs[:15], 1):
        v = "diverged" if c["val"] is None else f"{c['val']:.4f}" + (
            f" ± {c['val_std']:.4f}" if c["val_std"] is not None else "")
        print(f"{k:>4}  {c['label']:<45} {v:>17}  {R.fmt(c['bpc'], '{:.3f}'):>6}  {c['n_seeds']}")
    single = [c["id"] for c in configs[:5] if c["n_seeds"] < 3]
    if single:
        print(f"\nTop-5 configs with fewer than 3 seeds: {', '.join(single)}. Confirm them with:")
        print(f"  python scripts/run_experiments.py --ids {','.join(single)} --seeds 1,2,3")
    print(f"\nwrote {dash}, {args.out}/runs.csv, {args.out}/configs.csv, {args.out}/charts/")


if __name__ == "__main__":
    main()
