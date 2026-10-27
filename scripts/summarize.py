"""Collect every number needed for a writeup into one Markdown file.

  python scripts/summarize.py                     # -> results/summary.md + results/summary.json

Sections (A-D follow the metric plan; E-K are the analysis)
  Setup   data, model size, budget, learning rates, run counts
  A       every training run: train/val loss, perplexity, LR, gradient norm, clipping, time, chars
  B       every experiment report: best val + step, BPC, top-1, gap, params, memory, time to target
  C       confirming the best: seeds/mean/std, full validation split, curves, diversity, samples
  D       final models: test set, benchmark, human ratings, leakage (from final_eval.py), speed
  E-K     leaderboard with p-values, stage 1, controlled pairs, top cluster, telemetry, sweeps, n-grams

Statistics: Welch's t-test between two configs' per-seed final validation
losses (two-sided). With 3 seeds per config the test is weak: "not
significant" means "not shown", not "no effect". Many pairs are tested, so
treat p-values near 0.05 as suggestive rather than conclusive.
"""
import argparse
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import report as R  # noqa: E402

LN2 = math.log(2)
BASE = "01"

STAGE1 = {
    "optimizer": ["02", "03", "04", "05", "06", "07"],
    "activation": ["08", "09", "10", "11", "12"],
    "norm": ["13", "14", "15", "16"],
    "positional": ["17", "18"],
}

# (question, [(from_id, to_id, context)]): each pair differs in exactly one component
PAIRS = [
    ("RoPE instead of learned positions", [("01", "18", "AdamW, GELU, LayerNorm"),
                                            ("05", "51", "NorMuon, GELU, LayerNorm"),
                                            ("41", "21", "NorMuon, SwiGLU, RMSNorm")]),
    ("Sinusoidal instead of learned positions", [("01", "17", "AdamW, GELU, LayerNorm"),
                                                  ("41", "42", "NorMuon, SwiGLU, RMSNorm")]),
    ("Sinusoidal instead of RoPE", [("21", "42", "NorMuon, SwiGLU, RMSNorm"),
                                     ("19", "43", "AdamW, SwiGLU, RMSNorm")]),
    ("Removing normalization", [("01", "15", "AdamW, GELU, learned"),
                                 ("19", "40", "AdamW, SwiGLU, RoPE"),
                                 ("20", "38", "Muon, SwiGLU, RoPE"),
                                 ("21", "39", "NorMuon, SwiGLU, RoPE")]),
    ("RMSNorm instead of LayerNorm", [("01", "13", "AdamW, GELU, learned"),
                                       ("44", "19", "AdamW, SwiGLU, RoPE"),
                                       ("45", "21", "NorMuon, SwiGLU, RoPE")]),
    ("ScaleNorm instead of RMSNorm", [("19", "36", "AdamW, SwiGLU, RoPE"),
                                       ("21", "37", "NorMuon, SwiGLU, RoPE")]),
    ("DyT instead of RMSNorm/LayerNorm", [("01", "14", "AdamW, GELU, learned (vs LayerNorm)"),
                                           ("19", "32", "AdamW, SwiGLU, RoPE"),
                                           ("20", "33", "Muon, SwiGLU, RoPE"),
                                           ("21", "34", "NorMuon, SwiGLU, RoPE"),
                                           ("22", "35", "Lion, SwiGLU, RoPE")]),
    ("SwiGLU instead of GELU", [("01", "12", "AdamW, LayerNorm, learned"),
                                 ("51", "45", "NorMuon, LayerNorm, RoPE")]),
    ("GeGLU instead of SwiGLU", [("19", "26", "AdamW, RMSNorm, RoPE"),
                                  ("20", "30", "Muon, RMSNorm, RoPE"),
                                  ("21", "31", "NorMuon, RMSNorm, RoPE")]),
    ("Squared ReLU instead of SwiGLU", [("19", "27", "AdamW, RMSNorm, RoPE"),
                                         ("20", "28", "Muon, RMSNorm, RoPE"),
                                         ("21", "29", "NorMuon, RMSNorm, RoPE")]),
    ("SiLU instead of GELU", [("01", "09", "AdamW, LayerNorm, learned")]),
    ("Optimizer vs AdamW, classic stack (GELU, LayerNorm, learned)",
     [("01", x, "") for x in ("02", "03", "04", "05", "06", "07")]),
    ("Optimizer vs AdamW, modern stack (SwiGLU, RMSNorm, RoPE)",
     [("19", x, "") for x in ("20", "21", "22", "23", "24", "25")]),
    ("NorMuon instead of Muon", [("04", "05", "GELU, LayerNorm, learned"),
                                  ("20", "21", "SwiGLU, RMSNorm, RoPE"),
                                  ("38", "39", "SwiGLU, no norm, RoPE"),
                                  ("33", "34", "SwiGLU, DyT, RoPE")]),
]


# ------------------------------------------------------------------ stats
def _betacf(a, b, x):
    """Continued fraction for the regularized incomplete beta (Numerical Recipes)."""
    tiny, qab, qap, qam = 1e-300, a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return h


def betainc(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbeta) * _betacf(a, b, x) / a
    return 1.0 - math.exp(lbeta) * _betacf(b, a, 1 - x) / b


def welch(x, y):
    """Two-sided Welch t-test. Returns (t, df, p) or None if not testable."""
    if len(x) < 2 or len(y) < 2:
        return None
    mx, my = st.mean(x), st.mean(y)
    vx, vy = st.variance(x) / len(x), st.variance(y) / len(y)
    se2 = vx + vy
    if se2 <= 0:
        return None
    t = (my - mx) / math.sqrt(se2)
    df = se2 ** 2 / ((vx ** 2 / (len(x) - 1)) + (vy ** 2 / (len(y) - 1)))
    p = betainc(df / 2, 0.5, df / (df + t * t))
    return t, df, p


def verdict(delta, test):
    if test is None:
        return "needs ≥2 seeds each"
    p = test[2]
    if p < 0.05:
        return "**better**" if delta < 0 else "**worse**"
    if p < 0.10:
        return "suggestive " + ("(better)" if delta < 0 else "(worse)")
    return "not significant"


# ------------------------------------------------------------------ helpers
def fmt(v, f="{:.4f}", none="—"):
    return none if v is None or (isinstance(v, float) and not math.isfinite(v)) else f.format(v)


def pfmt(test):
    if test is None:
        return "—"
    p = test[2]
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def seed_vals(c):
    return [R.run_value(r, "val") for r in c["runs"] if R.run_value(r, "val") is not None]


def ms(c):
    if c is None or c["val"] is None:
        return "diverged" if c else "—"
    s = f"{c['val']:.4f}"
    return s + (f" ± {c['val_std']:.4f}" if c["val_std"] is not None else "")


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def last_telemetry(run):
    rows = [m for m in run["m"] if m.get("telemetry")]
    if not rows:
        return None
    last = rows[-1]
    t = last["telemetry"]
    out = {}
    ffn = [x for x in t.get("ffn", []) if x]
    if ffn:
        out["dead"] = st.mean(x["dead_frac"] for x in ffn)
        out["sparsity"] = st.mean(x["sparsity"] for x in ffn)
    att = [x for x in t.get("attn", []) if x and x.get("attn_entropy_per_head")]
    if att:
        heads = [h for x in att for h in x["attn_entropy_per_head"]]
        out["attn_entropy"] = st.mean(heads)
        out["attn_entropy_min"] = min(heads)
    if t.get("resid_norm_per_layer"):
        out["resid_last"] = t["resid_norm_per_layer"][-1]
    g = last.get("groups", {})
    if "block0" in g:
        out["upd_block0"] = g["block0"]["update_ratio"]
    lca = run["s"].get("lca") or {}
    tot = sum(lca.values())
    if tot:
        blocks = sum(v for k, v in lca.items() if k.startswith("block"))
        out["lca_blocks_share"] = blocks / tot
        out["lca_embed_share"] = lca.get("embed", 0.0) / tot
    return out


def mean_tele(c):
    ts = [x for x in (last_telemetry(r) for r in c["runs"] if not r["s"].get("diverged")) if x]
    keys = {k for t in ts for k in t}
    return {k: st.mean(t[k] for t in ts if k in t) for k in keys} if ts else {}


def sweep_results(root):
    out = {}
    for sp in glob.glob(os.path.join(root, "sweeps*", "*", "summary.json")):
        d = os.path.dirname(sp)
        name = os.path.basename(d)
        if "_lr" not in name:
            continue
        key, lr = name.rsplit("_lr", 1)
        s = R._json(sp) or {}
        c = R._json(os.path.join(d, "config.json")) or {}
        size = f"{c.get('n_layer', '?')}L×{c.get('d_model', '?')}"
        v = s.get("final_val_loss")
        out.setdefault((os.path.basename(os.path.dirname(d)), key, size), []).append(
            (float(lr), None if s.get("diverged") or v is None or not math.isfinite(v) else v))
    return out


# ------------------------------------------------------------------ per-run values
def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def run_metric(run, key):
    """Per-run values for the A-D tables (None if unavailable or diverged)."""
    s, m, c = run["s"], run["m"], run["cfg"]
    ok = not s.get("diverged")
    logged = [x for x in m if _num(x.get("grad_norm")) is not None]
    fe = s.get("final_eval") or {}
    inf = fe.get("inference") or {}
    ts = fe.get("text_stats") or {}
    get = {
        # A. every training run
        "train_loss": lambda: _num(s.get("final_train_ema")),
        "val_loss": lambda: _num(s.get("final_val_loss")) if ok else None,
        "ppl": lambda: (_num(s.get("final_ppl")) or (math.exp(s["final_val_loss"]) if _num(s.get("final_val_loss")) else None)) if ok else None,
        "peak_lr": lambda: _num(c.get("lr")),
        "grad_mean": lambda: st.mean(x["grad_norm"] for x in logged) if logged else None,
        "grad_final": lambda: logged[-1]["grad_norm"] if logged else None,
        "clip": lambda: _num(s.get("clip_rate")),
        "train_min": lambda: (_num(s.get("train_time_s")) or _num(s.get("wall_time_s")) or 0) / 60 or None,
        "chars": lambda: _num(s.get("tokens")),
        # B. every experiment report
        "best_val": lambda: _num(s.get("best_val_loss")) if ok else None,
        "best_step": lambda: _num(R._best_step(s)) if ok else None,
        "bpc": lambda: _num(s.get("final_val_loss")) / LN2 if ok and _num(s.get("final_val_loss")) else None,
        "acc": lambda: _num(s.get("final_acc")) if ok else None,
        "gap": lambda: _num(s.get("generalization_gap")) if ok else None,
        "params": lambda: _num(s.get("n_params")),
        "peak_mem": lambda: _num(s.get("peak_mem_mb")),
        # C. confirmation
        "full_bpc": lambda: _num((fe.get("full_val") or {}).get("bpc")),
        "full_acc": lambda: _num((fe.get("full_val") or {}).get("acc")),
        "distinct_1": lambda: _num(ts.get("distinct_1")),
        "distinct_2": lambda: _num(ts.get("distinct_2")),
        "distinct_3": lambda: _num(ts.get("distinct_3")),
        "rep_4": lambda: _num(ts.get("rep_4")),
        "real_word": lambda: _num(ts.get("real_word")),
        "story": lambda: _num((fe.get("ending_choice") or {}).get("acc")),
        "story_raw": lambda: _num((fe.get("ending_choice") or {}).get("acc_raw")),
        # D. inference (measured per run)
        "gen_ms": lambda: _num(inf.get("generate_ms_per_char")),
        "batch_cps": lambda: _num(inf.get("batch_chars_per_s")),
        "single_ms": lambda: _num(inf.get("single_ms")),
    }[key]
    try:
        return get()
    except (TypeError, ZeroDivisionError):
        return None


def agg(c, key):
    vals = [v for v in (run_metric(r, key) for r in c["runs"]) if v is not None]
    return st.mean(vals) if vals else None


def ttt(c, target):
    vals = [x["train_s"] / 60 for x in ((r["s"].get("time_to_target") or {}).get(str(target)) for r in c["runs"])
            if x and x.get("train_s") is not None]
    if not vals:
        return "—"
    out = f"{st.mean(vals):.1f}"
    return out + (f" ({len(vals)}/{c['n_seeds']})" if len(vals) < c["n_seeds"] else "")


def curve_points(c, steps):
    """Mean validation loss across seeds at (the eval nearest to) each step."""
    out = []
    for target in steps:
        vals = []
        for r in c["runs"]:
            ev = [e for e in r["s"].get("evals", []) if _num(e[1]) is not None]
            if ev:
                vals.append(min(ev, key=lambda e: abs(e[0] - target))[1])
        out.append(st.mean(vals) if vals else None)
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="results/runs")
    ap.add_argument("--out", default="results")
    ap.add_argument("--lrs", default="experiments/lrs.json")
    ap.add_argument("--samples-top", type=int, default=5, help="configs whose generation samples are included")
    args = ap.parse_args()

    runs = R.load_runs(args.runs)
    if not runs:
        raise SystemExit(f"no finished runs in {args.runs}")
    configs = R.aggregate(runs)
    by = {c["id"]: c for c in configs}
    base = by.get(BASE)
    ranked = [c for c in configs if c["val"] is not None]
    order = ranked + [c for c in configs if c["val"] is None]
    bl = R._baselines_for(runs[0]["cfg"]["data_dir"])
    J = {"configs": {}, "pairs": [], "stage1": {}}
    md = []

    def delta_vs(a, b):
        if a is None or b is None or a["val"] is None or b["val"] is None:
            return None, None
        return b["val"] - a["val"], welch(seed_vals(a), seed_vals(b))

    def lab(c):
        return c["label"] + ("" if c["val"] is not None else " ⚠ diverged")

    # ---------------------------------------------------------------- setup
    c0 = runs[0]["cfg"]
    meta = R._json(os.path.join(c0["data_dir"], "meta.json")) or {}
    s0 = runs[0]["s"]
    diverged = [r["name"] for r in runs if r["s"].get("diverged")]
    lrs = {k: v for k, v in (R._json(args.lrs) or {}).items() if not k.startswith("_")}
    md.append("# Experiment summary\n")
    md.append("Sections A–D follow the metric plan: A = every training run, B = every experiment report, "
              "C = confirming the best configurations, D = final model evaluation. After that: analysis "
              "(stage 1, controlled pairs, top cluster, telemetry, LR sweeps). All tables list configs best "
              "first; values are means over seeds.\n")
    md.append("## Setup\n")
    md.append(table(["item", "value"], [
        ["data", f"{c0['data_dir']} (char-level, vocab {len(meta.get('chars', [])) or '?'}; "
                 f"train {meta.get('train_tokens', 0):,} / val {meta.get('val_tokens', 0):,} / "
                 f"test {meta.get('test_tokens', 0):,} chars)"],
        ["model", f"{c0['n_layer']} layers, width {c0['d_model']}, {c0['n_head']} heads, context {c0['ctx']}, "
                  f"~{s0.get('n_params', 0):,} params (baseline)"],
        ["training", f"{c0['max_steps']} steps × batch {c0['batch_size']} × {c0['ctx']} = "
                     f"{c0['max_steps'] * c0['batch_size'] * c0['ctx']:,} chars per run; "
                     f"{c0.get('schedule', 'cosine')} schedule, warmup {c0.get('warmup')}, "
                     f"weight decay {c0.get('weight_decay')}, grad clip {c0.get('grad_clip')}"],
        ["evaluation", f"final validation loss on {c0.get('eval_batches', 20)} fixed batches (identical for every run)"],
        ["runs", f"{len(runs)} runs, {len(configs)} configs, {len(diverged)} diverged"
                 + (f" ({', '.join(diverged)})" if diverged else "")],
        ["baseline (01)", ms(base)],
    ]))
    if lrs:
        md.append("\nLearning rates (`config:<id>` = per-config override):\n\n")
        md.append(table(["optimizer / config", "lr"], [[k, f"{v:g}"] for k, v in lrs.items()]))

    # ---------------------------------------------------------------- A
    md.append("\n## A. Every training run\n")
    md.append("train loss = smoothed training loss at the end (EMA); val loss on the fixed validation batches; "
              "perplexity = e^val loss; peak LR = learning rate after warmup; gradient norm = before clipping, "
              "mean over training / last logged value; clipped = share of steps where clipping was applied; "
              "train time excludes evaluation; chars = characters processed.\n\n")
    rows = []
    for c in order:
        rows.append([lab(c), fmt(agg(c, "train_loss")), ms(c), fmt(agg(c, "ppl"), "{:.3f}"),
                     fmt(agg(c, "peak_lr"), "{:g}"),
                     fmt(agg(c, "grad_mean"), "{:.3f}") + " / " + fmt(agg(c, "grad_final"), "{:.3f}"),
                     fmt(agg(c, "clip"), "{:.1%}"), fmt(agg(c, "train_min"), "{:.1f}"),
                     fmt(agg(c, "chars"), "{:,.0f}")])
    md.append(table(["config", "train loss", "val loss", "perplexity", "peak LR", "grad norm mean / last",
                     "clipped", "train min", "chars"], rows))

    # ---------------------------------------------------------------- B
    targets = sorted({float(t) for r in runs for t in (r["s"].get("time_to_target") or {})}, reverse=True)
    md.append("\n## B. Every experiment report\n")
    md.append("best val = lowest validation loss at any evaluation, and its step; BPC = val loss / ln 2; top-1 = "
              "next-character accuracy; gap = val loss − loss on fixed training batches (positive = worse on "
              "unseen text); peak memory = process memory after backward (all runs share one Python process, so "
              "treat it as approximate); time to target = train minutes until validation loss first reached the "
              "target (n/m = only n of m seeds reached it).\n\n")
    rows = []
    for c in order:
        row = [lab(c), fmt(agg(c, "best_val")) + " @ " + fmt(agg(c, "best_step"), "{:.0f}"),
               fmt(agg(c, "bpc"), "{:.3f}"), fmt(agg(c, "acc"), "{:.1%}"), fmt(agg(c, "gap"), "{:+.4f}"),
               fmt(agg(c, "params"), "{:,.0f}"), fmt(agg(c, "peak_mem"), "{:,.0f}")]
        row += [ttt(c, t) for t in targets]
        rows.append(row)
    md.append(table(["config", "best val @ step", "BPC", "top-1", "gap", "params", "peak MB"]
                    + [f"min → {t}" for t in targets], rows))

    # ---------------------------------------------------------------- C
    md.append("\n## C. Confirming the best configurations\n")
    md.append("### C1. Seeds, mean and standard deviation\n\n")
    rows = [[lab(c), c["n_seeds"], ms(c), fmt(c["val_std"] / c["val"] if c["val"] and c["val_std"] else None, "{:.2%}"),
             ", ".join(fmt(R.run_value(r, "val"), "{:.4f}", "div.") for r in sorted(c["runs"], key=lambda r: r["cfg"]["seed"]))]
            for c in order if c["n_seeds"] >= 2]
    md.append(table(["config", "seeds", "val loss mean ± std", "relative std", "per seed"], rows) if rows
              else "No config has 2+ seeds yet.\n")

    md.append("\n### C2. Full validation-split evaluation\n\n")
    md.append("Every character of the validation split (not just the 20 fixed batches).\n\n")
    rows = [[lab(c), fmt(agg(c, "full_bpc"), "{:.3f}"), fmt(agg(c, "bpc"), "{:.3f}"),
             fmt((agg(c, "full_bpc") or 0) - (agg(c, "bpc") or 0) if agg(c, "full_bpc") and agg(c, "bpc") else None, "{:+.3f}"),
             fmt(agg(c, "full_acc"), "{:.1%}")] for c in order]
    md.append(table(["config", "full-split BPC", "fixed-batch BPC", "difference", "full-split top-1"], rows))

    md.append("\n### C3. Validation loss curves\n\n")
    max_steps = c0["max_steps"]
    steps = sorted({0, max_steps // 10, max_steps // 4, max_steps // 2, 3 * max_steps // 4, max_steps - 1})
    md.append("Mean validation loss at points of training (nearest evaluation). Charts: "
              "`results/charts/top_curves.png` and `results/charts/curves_*.png` (from scripts/analyze.py).\n\n")
    rows = [[lab(c)] + [fmt(v) for v in curve_points(c, steps)] for c in order]
    md.append(table(["config"] + [f"step {s}" for s in steps], rows))

    md.append("\n### C4. Repetition and diversity of generated text\n\n")
    md.append("From 5 fixed prompts × 300 characters per run. distinct-n = unique word n-grams / all (higher = more "
              "varied); rep-4 = share of word 4-grams repeating earlier in the same sample (higher = looping); real "
              "words = share of generated words found in the training text; story-ending = picking the real next "
              "sentence out of 4, calibrated / raw (chance 25%).\n\n")
    rows = [[lab(c), fmt(agg(c, "distinct_1"), "{:.3f}"), fmt(agg(c, "distinct_2"), "{:.3f}"),
             fmt(agg(c, "distinct_3"), "{:.3f}"), fmt(agg(c, "rep_4"), "{:.3f}"), fmt(agg(c, "real_word"), "{:.1%}"),
             fmt(agg(c, "story"), "{:.1%}") + " / " + fmt(agg(c, "story_raw"), "{:.1%}")] for c in order]
    md.append(table(["config", "distinct-1", "distinct-2", "distinct-3", "rep-4", "real words", "story-ending cal / raw"],
                    rows))

    md.append(f"\n### C5. Fixed-prompt generation samples (top {args.samples_top} configs and baseline, seed 1)\n\n")
    md.append("Same prompts, sampling seeds and settings (temperature 0.8, top-k 20) for every run.\n")
    pick = ranked[:args.samples_top] + ([base] if base and base not in ranked[:args.samples_top] else [])
    for c in pick:
        r = min(c["runs"], key=lambda r: r["cfg"]["seed"])
        smp = (r["s"].get("final_eval") or {}).get("samples") or []
        if not smp:
            continue
        md.append(f"\n**{c['label']}**\n")
        for x in smp:
            text = (x["prompt"] + x["text"]).replace("\n", " ⏎ ")
            md.append(f"\n> {text}\n")

    # ---------------------------------------------------------------- D
    md.append("\n## D. Final model evaluation\n")
    finals = [(r, R._json(os.path.join(r["dir"], "final_eval.json"))) for r in runs]
    finals = [(r, f) for r, f in finals if f]
    md.append("### D1. Held-out test set, task benchmark, human ratings, leakage\n\n")
    if finals:
        rows = []
        for r, f in finals:
            h = f.get("human") or {}
            mem = f.get("memorization") or {}
            rows.append([r["name"], fmt(R._get(f, "test", "loss")), fmt(R._get(f, "test", "bpc"), "{:.3f}"),
                         fmt(R._get(f, "test", "acc"), "{:.1%}"), fmt(R._get(f, "val_full", "bpc"), "{:.3f}"),
                         fmt(R._get(f, "ending_choice_test", "acc"), "{:.1%}"),
                         fmt(_num(h.get("fluency")), "{:.2f}") + " / " + fmt(_num(h.get("coherence")), "{:.2f}")
                         + (f" (n={h['n']})" if h else ""),
                         fmt(_num(mem.get("frac_64")), "{:.1%}"), fmt(_num(mem.get("longest_match_max")), "{:.0f}")])
        md.append(table(["run", "test loss", "test BPC", "test top-1", "full-val BPC", "story-ending (test)",
                         "human fluency / coherence", "gen. 64-char windows in train", "longest verbatim match"], rows))
        leak = R._json(os.path.join(c0["data_dir"], "leakage.json"))
        if leak:
            md.append("\nDataset leakage: share of val/test windows that also occur verbatim in the training text.\n\n")
            md.append(table(["split", "32 chars", "64 chars", "128 chars"],
                            [[s] + [fmt(_num((leak.get(s) or {}).get(L)), "{:.2%}") for L in ("32", "64", "128")]
                             for s in ("val", "test") if s in leak]))
    else:
        md.append("Not run yet. After choosing the final models: `python scripts/rate_samples.py <runs>` for blind "
                  "human ratings, then `python scripts/final_eval.py <runs>` for the test set, story-ending on test, "
                  "speed and leakage checks. Rerun this script afterwards and this section fills in.\n")
    md.append("\n### D2. Inference latency and throughput (every run)\n\n")
    md.append("Measured on your machine at the end of each run: generation = ms per generated character (batch 1); "
              "forward = one full-context forward pass for 1 sequence / chars per second for a batch of 32.\n\n")
    rows = [[lab(c), fmt(agg(c, "gen_ms"), "{:.1f}"), fmt(agg(c, "single_ms"), "{:.1f}"),
             fmt(agg(c, "batch_cps"), "{:,.0f}")] for c in order]
    md.append(table(["config", "generation ms/char", "forward ms (1 seq)", "batch chars/s"], rows))

    # ---------------------------------------------------------------- analysis
    md.append("\n# Analysis\n")
    md.append("\n## E. Leaderboard with significance vs baseline\n")
    md.append("Δ = difference in final validation loss vs baseline 01 (negative = better). p = Welch t-test.\n\n")
    rows = []
    for i, c in enumerate(configs, 1):
        d, t = delta_vs(base, c) if c["id"] != BASE else (None, None)
        rows.append([i if c["val"] is not None else "—", c["label"], c["n_seeds"], ms(c), fmt(c["bpc"], "{:.3f}"),
                     fmt(d, "{:+.4f}"), pfmt(t)])
        J["configs"][c["id"]] = {"label": c["label"], "seeds": seed_vals(c), "mean": c["val"], "std": c["val_std"],
                                 "bpc": c["bpc"], "delta_vs_baseline": d, "p_vs_baseline": t[2] if t else None,
                                 **{k: agg(c, k) for k in ("train_loss", "ppl", "peak_lr", "grad_mean", "clip",
                                                            "train_min", "best_val", "best_step", "acc", "gap",
                                                            "full_bpc", "distinct_2", "rep_4", "real_word", "story",
                                                            "gen_ms", "batch_cps")}}
    md.append(table(["rank", "config", "seeds", "val loss", "BPC", "Δ vs 01", "p"], rows))

    md.append("\n## F. Stage 1: one change at a time vs baseline 01\n")
    for dim, ids in STAGE1.items():
        rows = [[f"{base[dim]} (baseline)", base["n_seeds"], ms(base), "—", "—", "—"]] if base else []
        for i in ids:
            c = by.get(i)
            if not c:
                continue
            d, t = delta_vs(base, c)
            rows.append([c[dim], c["n_seeds"], ms(c), fmt(d, "{:+.4f}"), pfmt(t), verdict(d, t) if d is not None else "diverged"])
            J["stage1"].setdefault(dim, []).append({"id": i, "variant": c[dim], "delta": d, "p": t[2] if t else None})
        rows.sort(key=lambda r: (r[2] in ("—", "diverged"), r[2]))
        md.append(f"\n### {dim}\n\n" + table([dim, "seeds", "val loss", "Δ vs 01", "p", "verdict"], rows))

    md.append("\n## G. Controlled pairs: the same change in different contexts\n")
    md.append("Each row changes exactly one component; comparing rows within a question shows interactions.\n")
    for q, pairs in PAIRS:
        rows = []
        for a_id, b_id, ctx in pairs:
            a, b = by.get(a_id), by.get(b_id)
            if not a or not b:
                continue
            d, t = delta_vs(a, b)
            rows.append([ctx or b["optimizer"], f"{a_id} → {b_id}", ms(a), ms(b), fmt(d, "{:+.4f}"), pfmt(t),
                         verdict(d, t) if d is not None else "diverged"])
            J["pairs"].append({"question": q, "from": a_id, "to": b_id, "context": ctx, "delta": d,
                               "p": t[2] if t else None})
        if rows:
            md.append(f"\n### {q}\n\n" + table(["context", "configs", "before", "after", "Δ", "p", "verdict"], rows))

    md.append("\n## H. Top cluster vs #1\n\n")
    if ranked:
        top = ranked[0]
        rows = []
        for c in ranked[:12]:
            d, t = delta_vs(top, c) if c is not top else (None, None)
            rows.append([c["label"], c["n_seeds"], ms(c), fmt(d, "{:+.4f}"), pfmt(t),
                         "—" if c is top else verdict(d, t).replace("worse", "worse than #1")])
        md.append(table(["config", "seeds", "val loss", "Δ vs #1", "p", "verdict"], rows))

    md.append("\n## I. Telemetry at the end of training (why configs differ)\n")
    md.append("dead = FFN units never active on a batch; sparsity = activations ≤ 0; attention entropy in nats "
              "(mean / lowest head; low = sharp or collapsed heads); resid = residual-stream norm after the last "
              "block; upd = ‖ΔW‖/‖W‖ of block 0 on the last step; LCA share = share of the total first-order loss "
              "reduction made by the transformer blocks / the embeddings.\n\n")
    rows = []
    for c in ranked:
        t = mean_tele(c)
        J["configs"][c["id"]]["telemetry"] = t
        rows.append([c["label"], fmt(t.get("dead"), "{:.1%}"), fmt(t.get("sparsity"), "{:.1%}"),
                     fmt(t.get("attn_entropy"), "{:.2f}") + " / " + fmt(t.get("attn_entropy_min"), "{:.2f}"),
                     fmt(t.get("resid_last"), "{:.2f}"), fmt(t.get("upd_block0"), "{:.2e}"),
                     fmt(t.get("lca_blocks_share"), "{:.0%}") + " / " + fmt(t.get("lca_embed_share"), "{:.0%}")])
    md.append(table(["config", "dead", "sparsity", "attn entropy", "resid", "upd block0", "LCA blocks / embed"], rows))

    sw = sweep_results(os.path.dirname(os.path.normpath(args.runs)))
    if sw:
        md.append("\n## J. Learning-rate sweeps\n")
        md.append("Short runs (500 steps by default); value = final validation loss. ★ = best.\n\n")
        rows = []
        for (folder, key, size), pts in sorted(sw.items()):
            pts.sort()
            ok = [p for p in pts if p[1] is not None]
            best = min(ok, key=lambda p: p[1])[0] if ok else None
            rows.append([folder, key, size, ", ".join(
                f"{lr:g}: " + ("div." if v is None else f"{v:.4f}") + (" ★" if lr == best else "") for lr, v in pts)])
        md.append(table(["folder", "optimizer / config", "model", "lr: val loss"], rows))

    if bl:
        md.append("\n## K. n-gram reference models (bits per character)\n\n")
        splits = [s for s in ("val", "test") if s in bl]
        md.append(table(["order"] + splits, [[f"{k}-gram"] + [fmt(bl[s].get(k), "{:.3f}") for s in splits]
                                             for k in sorted(bl[splits[0]], key=int)]))
        if ranked:
            o, v = min(((int(k), x) for k, x in bl["val"].items()), key=lambda t: t[1])
            md.append(f"\nBest n-gram ({o}-gram): {v:.3f} BPC. Best transformer config: {ranked[0]['label']}, "
                      f"{ranked[0]['bpc']:.3f} BPC ({ranked[0]['bpc'] - v:+.3f}).\n")

    md.append("\n---\nStatistics: Welch's t-test on per-seed final validation losses, two-sided. With 3 seeds the "
              "test has little power: 'not significant' means the difference wasn't shown, not that there is none. "
              "Many comparisons are made, so expect roughly 1 in 20 false 'significant' results among true "
              "zero effects; weigh p-values near 0.05 accordingly.\n")

    os.makedirs(args.out, exist_ok=True)
    path_md = os.path.join(args.out, "summary.md")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(J, f, indent=2)
    print(f"wrote {path_md} and {os.path.join(args.out, 'summary.json')}")


if __name__ == "__main__":
    main()
