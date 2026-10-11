"""Run experiment configs back to back. Resumable: finished runs are skipped.

  python scripts/run_experiments.py                       # all 60 configs (~106 runs)
  python scripts/run_experiments.py --ids 01-18           # stage 1 only
  python scripts/run_experiments.py --ids 01-18 --dry-run # just list what would run
  python scripts/run_experiments.py --ids 05 --seeds 1,2,3  # extra seeds for one config
  python scripts/run_experiments.py --set d_model=128 n_layer=4 --out results_big
"""
import argparse
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import experiments as E  # noqa: E402
from nllm.report import build_dashboard  # noqa: E402
from nllm.train import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="experiments/configs.json")
    ap.add_argument("--lrs", default="experiments/lrs.json")
    ap.add_argument("--out", default="results/runs")
    ap.add_argument("--ids", default="", help="e.g. 01-18,R01-R10 (default: all)")
    ap.add_argument("--seeds", default="", help="override seeds, e.g. 1,2,3")
    ap.add_argument("--set", nargs="*", default=[], help="override config keys, e.g. max_steps=500")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--dashboard", default=None,
                    help="dashboard path (default: <parent of --out>/dashboard.html)")
    args = ap.parse_args()

    exp = E.load(args.configs)
    lrs = E.load_lrs(args.lrs)
    overrides = E.parse_overrides(args.set)
    by_id = {c["id"]: c for c in exp["configs"]}
    ids = E.parse_ids(args.ids, list(by_id))

    jobs = []
    for i in ids:
        entry = by_id[i]
        seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else entry["seeds"]
        for seed in seeds:
            name = E.run_name(entry, seed)
            cfg = E.resolve(exp, entry, lrs, {**overrides, "seed": seed})
            jobs.append((name, cfg, os.path.join(args.out, name)))

    todo = [j for j in jobs if not E.is_done(j[2])]
    print(f"{len(jobs)} runs selected, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    if args.dry_run:
        for name, cfg, _ in todo:
            print(f"  {name:<45} lr={cfg['lr']:g}")
        return

    t_start, failed = time.time(), []
    for k, (name, cfg, out_dir) in enumerate(todo, 1):
        print(f"\n[{k}/{len(todo)}] {name}  lr={cfg['lr']:g}")
        try:
            s = train(cfg, out_dir, verbose=not args.quiet)
            tag = "DIVERGED" if s["diverged"] else f"val {s['final_val_loss']:.4f}"
            print(f"  -> {tag}  ({s['wall_time_s'] / 60:.1f} min)")
        except KeyboardInterrupt:
            print("\ninterrupted; rerun the same command to resume")
            return
        except Exception:
            traceback.print_exc()
            failed.append(name)
        try:
            dash = args.dashboard or os.path.join(os.path.dirname(os.path.normpath(args.out)), "dashboard.html")
            build_dashboard(args.out, dash)
            print(f"  report: {os.path.join(out_dir, 'report.html')}   dashboard: {dash}")
        except Exception as err:
            print(f"  (dashboard not updated: {err!r})")
        elapsed = time.time() - t_start
        eta = elapsed / k * (len(todo) - k)
        print(f"  elapsed {elapsed / 60:.0f} min, ETA {eta / 60:.0f} min")
    if failed:
        print(f"\n{len(failed)} runs crashed: {failed}")


if __name__ == "__main__":
    main()
