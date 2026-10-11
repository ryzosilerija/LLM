"""HTML reports.

  run_report(run_dir)                  -> run_dir/report.html
  build_dashboard(runs_dir, out_html)  -> one page comparing every run

Charts are matplotlib PNGs embedded as base64, so each report is a single
file you can open in a browser or send to someone. Tables are sortable
(click a column header). Runs made before the extra metrics existed show
"—" for the metrics they don't have.
"""
import base64
import glob
import html
import io
import json
import math
import os
import statistics as st
import time

LN2 = math.log(2)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"

DIMENSIONS = {
    "optimizer": ["sgd", "adamw", "lion", "muon", "normuon", "sophia", "shampoo"],
    "activation": ["relu", "gelu", "silu", "sqrelu", "geglu", "swiglu"],
    "norm": ["layernorm", "rmsnorm", "dyt", "scalenorm", "none"],
    "positional": ["learned", "sinusoidal", "rope"],
}
STAGE1 = {"optimizer": ["01", "02", "03", "04", "05", "06", "07"],
          "activation": ["08", "01", "09", "10", "11", "12"],
          "norm": ["01", "13", "14", "16", "15"],
          "positional": ["01", "17", "18"]}
BASELINE_ID = "01"


# ----------------------------------------------------------------- loading
def _json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_metrics(d):
    path = os.path.join(d, "metrics.jsonl")
    if not os.path.exists(path):
        return []
    rows = []
    for line in open(path, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_run(d, with_metrics=True):
    s, c = _json(os.path.join(d, "summary.json")), _json(os.path.join(d, "config.json"))
    if s is None or c is None:
        return None
    name = os.path.basename(os.path.normpath(d))
    return {"dir": d, "name": name, "id": name.split("_")[0], "cfg": c, "s": s,
            "m": read_metrics(d) if with_metrics else []}


def load_runs(runs_dir, with_metrics=True):
    runs = [load_run(os.path.dirname(p), with_metrics) for p in glob.glob(os.path.join(runs_dir, "*", "summary.json"))]
    return sorted([r for r in runs if r], key=lambda r: r["name"])


# ----------------------------------------------------------------- metrics
def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v) if math.isfinite(v) else None


def _get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return _num(d)


def _best_step(s):
    if s.get("best_val_step") is not None:
        return s["best_val_step"]
    ev = [e for e in s.get("evals", []) if _num(e[1]) is not None]
    return min(ev, key=lambda e: e[1])[0] if ev else None


def _div(v, k):
    return None if v is None else v / k


def _pct(v):
    return f"{v:.1%}"


# key, label, getter(summary), format, better
METRICS = [
    ("val", "val loss", lambda s: _get(s, "final_val_loss"), "{:.4f}", "low"),
    ("bpc", "BPC", lambda s: _div(_get(s, "final_val_loss"), LN2), "{:.3f}", "low"),
    ("ppl", "perplexity", lambda s: _get(s, "final_ppl") or (
        math.exp(_get(s, "final_val_loss")) if _get(s, "final_val_loss") is not None else None), "{:.2f}", "low"),
    ("acc", "top-1 acc", lambda s: _get(s, "final_acc"), _pct, "high"),
    ("full_bpc", "full-val BPC", lambda s: _get(s, "final_eval", "full_val", "bpc"), "{:.3f}", "low"),
    ("gap", "gen. gap", lambda s: _get(s, "generalization_gap"), "{:+.4f}", None),
    ("best_step", "best step", lambda s: _num(_best_step(s)), "{:.0f}", None),
    ("story", "story-ending acc", lambda s: _get(s, "final_eval", "ending_choice", "acc"), _pct, "high"),
    ("real_word", "real words", lambda s: _get(s, "final_eval", "text_stats", "real_word"), _pct, "high"),
    ("distinct_2", "distinct-2", lambda s: _get(s, "final_eval", "text_stats", "distinct_2"), "{:.3f}", "high"),
    ("rep_4", "rep-4", lambda s: _get(s, "final_eval", "text_stats", "rep_4"), "{:.3f}", "low"),
    ("tok_s", "train chars/s", lambda s: _get(s, "train_tokens_per_s") or _get(s, "tokens_per_s"), "{:,.0f}", "high"),
    ("train_min", "train min", lambda s: _div(_get(s, "train_time_s") or _get(s, "wall_time_s"), 60), "{:.1f}", "low"),
    ("peak_mem", "peak MB", lambda s: _get(s, "peak_mem_mb"), "{:,.0f}", "low"),
    ("clip", "clip rate", lambda s: _get(s, "clip_rate"), _pct, None),
    ("params", "params", lambda s: _get(s, "n_params"), "{:,.0f}", None),
]
METRIC = {m[0]: m for m in METRICS}


def fmt(v, f, none="—"):
    if v is None:
        return none
    return f(v) if callable(f) else f.format(v)


def run_value(run, key):
    if run["s"].get("diverged") and key not in ("tok_s", "train_min", "peak_mem", "params", "clip"):
        return None
    return METRIC[key][2](run["s"])


def label(c):
    return f"{c['id']} {c['optimizer']}/{c['activation']}/{c['norm']}/{c['positional']}"


def aggregate(runs):
    """One row per config id: mean (and std) of each metric over seeds."""
    by = {}
    for r in runs:
        by.setdefault(r["id"], []).append(r)
    rows = []
    for cid, rs in by.items():
        c = rs[0]["cfg"]
        row = {"id": cid, "optimizer": c["optimizer"], "activation": c["activation"], "norm": c["norm"],
               "positional": c["positional"], "lr": c.get("lr"), "runs": rs, "n_seeds": len(rs),
               "n_diverged": sum(bool(r["s"].get("diverged")) for r in rs)}
        for key, *_ in METRICS:
            vals = [v for v in (run_value(r, key) for r in rs) if v is not None]
            row[key] = st.mean(vals) if vals else None
            row[key + "_std"] = st.stdev(vals) if len(vals) > 1 else None
            row[key + "_n"] = len(vals)
        row["label"] = label(row)
        rows.append(row)
    return sorted(rows, key=lambda r: (r["val"] is None, r["val"] if r["val"] is not None else 0))


def mean_curve(runs, field="val_loss"):
    curves = [[(m["step"], m[field]) for m in r["m"] if _num(m.get(field)) is not None] for r in runs]
    curves = [c for c in curves if c]
    if not curves:
        return []
    n = min(len(c) for c in curves)
    return [(curves[0][k][0], st.mean(c[k][1] for c in curves)) for k in range(n)]


# ------------------------------------------------------------------ charts
def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "axes.edgecolor": GRID, "xtick.color": INK2,
                         "ytick.color": INK2, "legend.frameon": False})
    return plt


def _style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid:
        ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9)


def _embed(fig, plt, alt, save_path=None):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(buf.getvalue())
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<figure><img src="data:image/png;base64,{b64}" alt="{html.escape(alt)}"></figure>'


def _ngram_line(ax, bl, orient="v"):
    if not bl or "val" not in bl:
        return
    order, v = min(((int(k), x) for k, x in bl["val"].items()), key=lambda t: t[1])
    if orient == "v":
        ax.axvline(v, color=INK2, lw=1, ls="--")
        ax.text(v, 1.0, f" best n-gram ({order}-gram) {v:.2f}", transform=ax.get_xaxis_transform(),
                fontsize=8, color=INK2, va="bottom", ha="left")
    else:
        ax.axhline(v, color=INK2, lw=1, ls="--")
        ax.text(1.0, v, f"{order}-gram {v:.2f} ", transform=ax.get_yaxis_transform(), fontsize=8,
                color=INK2, va="bottom", ha="right")


def chart_run_curves(plt, run, ref):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    _style(ax)
    m = run["m"]
    tr = [(r["step"], r["train_ema"]) for r in m if _num(r.get("train_ema")) is not None]
    va = [(r["step"], r["val_loss"]) for r in m if _num(r.get("val_loss")) is not None]
    te = [(r["step"], r["train_eval_loss"]) for r in m if _num(r.get("train_eval_loss")) is not None]
    if tr:
        ax.plot(*zip(*tr), color=PALETTE[0], lw=1.5, alpha=0.55, label="train loss (EMA)")
    if te:
        ax.plot(*zip(*te), color=PALETTE[0], lw=2, marker="o", ms=4, label="train loss (fixed batches)")
    if va:
        ax.plot(*zip(*va), color=PALETTE[1], lw=2, marker="o", ms=4, label="validation loss")
    for name, curve, ls in ref:
        if curve:
            ax.plot(*zip(*curve), color=INK2, lw=1.2, ls=ls, label=name)
    ys = [v for _, v in tr + va + te] + [v for _, c, _ in ref for _, v in c]
    if ys:
        ax.set_ylim(min(ys) - 0.05, min(max(ys), min(ys) + 2.5) + 0.05)
    ax.set_xlabel("step")
    ax.set_ylabel("loss (nats / char)")
    ax.set_title("Loss curves", loc="left", fontsize=11)
    ax.legend(fontsize=8.5)
    return _embed(fig, plt, "training and validation loss curves")


def chart_run_dynamics(plt, run):
    m = run["m"]
    panels = [("lr", "learning rate", None), ("grad_norm", "gradient norm (before clipping)",
                                               run["cfg"].get("grad_clip")),
              ("clip_rate", "share of steps clipped", None)]
    panels = [p for p in panels if any(_num(r.get(p[0])) is not None for r in m)]
    if not panels:
        return ""
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 3.1))
    axes = axes if len(panels) > 1 else [axes]
    for ax, (k, title, line) in zip(axes, panels):
        _style(ax)
        pts = [(r["step"], r[k]) for r in m if _num(r.get(k)) is not None]
        ax.plot(*zip(*pts), color=PALETTE[0], lw=2)
        if line:
            ax.axhline(line, color=INK2, ls="--", lw=1)
            ax.text(1.0, line, "clip threshold ", transform=ax.get_yaxis_transform(), fontsize=8,
                    color=INK2, ha="right", va="bottom")
        if k == "clip_rate":
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("step")
    fig.tight_layout()
    return _embed(fig, plt, "learning rate, gradient norm and clipping rate over training")


def _group_order(groups):
    blocks = sorted([g for g in groups if g.startswith("block")], key=lambda g: int(g[5:]))
    return [g for g in ["embed"] + blocks + ["final_norm", "head"] if g in groups]


def chart_run_layers(plt, run):
    m = [r for r in run["m"] if r.get("groups")]
    if not m:
        return ""
    groups = _group_order(m[-1]["groups"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    _style(a1)
    for i, g in enumerate(groups):
        pts = [(r["step"], r["groups"][g]["update_ratio"]) for r in m if g in r["groups"]]
        pts = [(s, v) for s, v in pts if v > 0]
        if pts:
            a1.plot(*zip(*pts), color=PALETTE[i % len(PALETTE)], lw=1.8, label=g)
    a1.set_yscale("log")
    a1.set_xlabel("step")
    a1.set_title("Update-to-weight ratio ‖ΔW‖/‖W‖ by layer group", loc="left", fontsize=10)
    a1.legend(fontsize=8, ncol=2)
    _style(a2, "x")
    lca = run["s"].get("lca", {})
    vals = [lca.get(g, 0.0) for g in groups]
    a2.barh(range(len(groups)), vals, color=[PALETTE[i % len(PALETTE)] for i in range(len(groups))], height=0.6)
    a2.set_yticks(range(len(groups)))
    a2.set_yticklabels(groups)
    a2.invert_yaxis()
    a2.axvline(0, color=INK2, lw=0.8)
    a2.set_title("Loss change allocation (negative = reduced loss)", loc="left", fontsize=10)
    fig.tight_layout()
    return _embed(fig, plt, "per-layer update ratios and loss change allocation")


def chart_rank(plt, configs, highlight=None, bl=None, metric="bpc", max_rows=30, save_path=None, title=None):
    rows = [c for c in configs if c.get(metric) is not None]
    if not rows:
        return ""
    if len(rows) > max_rows:
        keep = rows[:max_rows - 1]
        extra = [c for c in rows[max_rows - 1:] if c["id"] in (highlight, BASELINE_ID)]
        rows = keep + extra
    fig, ax = plt.subplots(figsize=(9, 0.27 * len(rows) + 1.3))
    _style(ax, "x")
    vals = [c[metric] for c in rows]
    errs = [c.get(metric + "_std") or 0 for c in rows]
    colors = [PALETTE[0] if c["id"] == highlight else (PALETTE[1] if c["id"] == BASELINE_ID else MUTED)
              for c in rows]
    if highlight is None:
        colors = [PALETTE[1] if c["id"] == BASELINE_ID else PALETTE[0] for c in rows]
    ax.barh(range(len(rows)), vals, xerr=errs, color=colors, height=0.62, error_kw={"ecolor": INK2, "lw": 0.8})
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{c['label']}  (n={c['n_seeds']})" for c in rows], fontsize=7.5)
    ax.invert_yaxis()
    lo = min(v - e for v, e in zip(vals, errs))
    hi = max(v + e for v, e in zip(vals, errs))
    if bl and "val" in bl:  # keep the n-gram reference line inside the axes
        nv = min(bl["val"].values())
        lo, hi = min(lo, nv), max(hi, nv)
    pad = 0.05 * (hi - lo) + 0.01
    ax.set_xlim(lo - pad, hi + pad)
    _ngram_line(ax, bl)
    ax.set_xlabel("validation bits per character (lower is better)")
    who = "this config in blue, " if highlight else ""
    ax.set_title(title or f"Configs ranked by final validation BPC ({who}baseline 01 in orange; bars = mean ± std over seeds)",
                 loc="left", fontsize=10, pad=18)
    return _embed(fig, plt, "configs ranked by validation bits per character", save_path)


def chart_top_curves(plt, configs, k=8, save_path=None):
    top = [c for c in configs if c["val"] is not None][:k]
    if not top:
        return ""
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _style(ax)
    for i, c in enumerate(sorted(top, key=lambda c: c["id"])):
        curve = mean_curve(c["runs"])
        if curve:
            ax.plot(*zip(*curve), color=PALETTE[i % 8], lw=2, label=c["label"])
    ys = [v for c in top for _, v in mean_curve(c["runs"])]
    if ys:
        ax.set_ylim(min(ys) - 0.03, min(max(ys), min(ys) + 1.5))
    ax.set_xlabel("step")
    ax.set_ylabel("validation loss")
    ax.set_title(f"Validation curves of the top {len(top)} configs (mean over seeds)", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    return _embed(fig, plt, "validation curves of the top configs", save_path)


def chart_scatter(plt, configs, xkey, xlabel, title, save_path=None, logx=False):
    pts = [c for c in configs if c.get(xkey) is not None and c.get("bpc") is not None]
    if len(pts) < 2:
        return ""
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _style(ax, "both")
    ax.scatter([c[xkey] for c in pts], [c["bpc"] for c in pts], s=36, color=PALETTE[0],
               edgecolor=SURFACE, linewidth=1.5, zorder=3)
    for c in sorted(pts, key=lambda c: c["bpc"])[:5]:
        ax.annotate(c["id"], (c[xkey], c["bpc"]), xytext=(4, 3), textcoords="offset points", fontsize=8,
                    color=INK2)
    if logx:
        ax.set_xscale("log")
    elif xkey == "tok_s":
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlabel(xlabel)
    ax.set_ylabel("validation BPC")
    ax.set_title(title, loc="left", fontsize=11)
    return _embed(fig, plt, title, save_path)


def chart_stage1(plt, configs, save_path=None):
    by = {c["id"]: c for c in configs}
    if not any(i in by for ids in STAGE1.values() for i in ids if i != BASELINE_ID):
        return ""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for ax, (dim, ids) in zip(axes.flat, STAGE1.items()):
        _style(ax, "x")
        present = [i for i in ids if i in by]
        names = [by[i][dim] for i in present]
        means = [by[i]["val"] for i in present]
        stds = [by[i]["val_std"] or 0 for i in present]
        colors = [PALETTE[DIMENSIONS[dim].index(n) % 8] for n in names]
        ax.barh(range(len(present)), [m if m is not None else 0 for m in means], xerr=stds, color=colors,
                height=0.6, error_kw={"ecolor": INK2, "lw": 1})
        for yi, (m, s) in enumerate(zip(means, stds)):
            ax.text((m + s if m is not None else 0) + 0.004, yi, " diverged" if m is None else f" {m:.3f}",
                    va="center", fontsize=8, color=INK2)
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        finite = [m for m in means if m is not None]
        if finite:
            ax.set_xlim(min(finite) - 0.08, max(finite) + 0.1)
        ax.set_title(f"{dim}: final validation loss", loc="left", fontsize=10)
    fig.suptitle("Stage 1: one change at a time vs baseline (mean ± std over seeds, lower is better)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    return _embed(fig, plt, "stage 1 comparison panels", save_path)


def chart_stage1_curves(plt, configs, save_path_fmt=None):
    by = {c["id"]: c for c in configs}
    out = []
    for dim, ids in STAGE1.items():
        present = [i for i in ids if i in by and mean_curve(by[i]["runs"])]
        if len(present) < 2:
            continue
        fig, ax = plt.subplots(figsize=(7.5, 4))
        _style(ax)
        for i in present:
            name = by[i][dim]
            ax.plot(*zip(*mean_curve(by[i]["runs"])), lw=2, color=PALETTE[DIMENSIONS[dim].index(name) % 8],
                    label=name)
        ys = [v for i in present for _, v in mean_curve(by[i]["runs"])]
        ax.set_ylim(min(ys) - 0.03, min(max(ys), min(ys) + 1.5))
        ax.set_xlabel("step")
        ax.set_ylabel("validation loss")
        ax.set_title(f"Validation loss by {dim} (stage 1, mean over seeds)", loc="left", fontsize=11)
        ax.legend(fontsize=9)
        out.append(_embed(fig, plt, f"validation curves by {dim}",
                          save_path_fmt.format(dim) if save_path_fmt else None))
    return "".join(out)


def chart_activation_telemetry(plt, configs, save_path=None):
    by = {c["id"]: c for c in configs}
    names, dead, sparse = [], [], []
    for i in STAGE1["activation"]:
        if i not in by:
            continue
        last = [r["m"][-1]["telemetry"]["ffn"] for r in by[i]["runs"] if r["m"] and r["m"][-1].get("telemetry")]
        if not last:
            continue
        names.append(by[i]["activation"])
        dead.append(st.mean(st.mean(l["dead_frac"] for l in L) for L in last))
        sparse.append(st.mean(st.mean(l["sparsity"] for l in L) for L in last))
    if len(names) < 2:
        return ""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, vals, title in ((axes[0], dead, "Dead FFN units (never active on a batch)"),
                            (axes[1], sparse, "Activation sparsity (output ≤ 0)")):
        _style(ax, "x")
        ax.barh(range(len(names)), vals, height=0.6,
                color=[PALETTE[DIMENSIONS["activation"].index(n)] for n in names])
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(title, loc="left", fontsize=10)
    fig.tight_layout()
    return _embed(fig, plt, "dead units and sparsity by activation", save_path)


def chart_update_ratio_by_optimizer(plt, configs, save_path=None):
    by = {c["id"]: c for c in configs}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    _style(ax)
    drawn = 0
    for i in STAGE1["optimizer"]:
        if i not in by:
            continue
        m = by[i]["runs"][0]["m"]
        pts = [(r["step"], r["groups"]["block0"]["update_ratio"]) for r in m if "block0" in r.get("groups", {})]
        pts = [(s, v) for s, v in pts if v > 0]
        if pts:
            name = by[i]["optimizer"]
            ax.plot(*zip(*pts), lw=2, color=PALETTE[DIMENSIONS["optimizer"].index(name)], label=name)
            drawn += 1
    if drawn < 2:
        plt.close(fig)
        return ""
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("‖ΔW‖ / ‖W‖  (block 0)")
    ax.set_title("Update-to-weight ratio by optimizer (stage 1, seed of first run)", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    return _embed(fig, plt, "update to weight ratio by optimizer", save_path)


# -------------------------------------------------------------------- html
CSS = """
body{font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:#fcfcfb;color:#0b0b0b;margin:0}
main{max-width:1150px;margin:0 auto;padding:24px 16px 72px}
h1{font-size:22px;margin:0 0 4px}
h2{font-size:17px;margin:36px 0 10px;padding-bottom:4px;border-bottom:1px solid #e4e3df}
h3{font-size:15px;margin:20px 0 6px}
a{color:#2a78d6}
.muted,.note{color:#52514e}.note{font-size:13px;max-width:820px}
.chips span{display:inline-block;background:#f0efec;border-radius:4px;padding:2px 8px;margin:2px 4px 2px 0;font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px;margin:12px 0}
.tile{background:#fff;border:1px solid #e4e3df;border-radius:6px;padding:8px 12px}
.tile .v{font-size:19px;font-weight:600;font-variant-numeric:tabular-nums;overflow-wrap:anywhere}.tile .v.t{font-size:15px}.tile .k{font-size:12px;color:#52514e}
.wrap{overflow-x:auto;margin:8px 0}
table{border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}
th,td{padding:4px 9px;border-bottom:1px solid #eeede9;text-align:right;white-space:nowrap}
th:first-child,td:first-child,th.l,td.l{text-align:left}
th{color:#52514e;font-weight:600;background:#fcfcfb}
table.sortable th{cursor:pointer}table.sortable th:hover{color:#0b0b0b}
tr.hl td{background:#eaf2fc}
figure{margin:12px 0}img{max-width:min(100%,1000px);height:auto}
pre{white-space:pre-wrap;background:#f4f3f0;padding:10px 12px;border-radius:6px;font-size:13px;margin:4px 0 12px}
.warn{background:#fdf3e6;border:1px solid #f1d3a8;border-radius:6px;padding:8px 12px;font-size:14px}
details summary{cursor:pointer;color:#52514e}
"""

SORT_JS = """
document.querySelectorAll('table.sortable').forEach(t=>{t.querySelectorAll('th').forEach((th,i)=>{
th.addEventListener('click',()=>{const tb=t.tBodies[0],rows=[...tb.rows],asc=th.dataset.asc!=='1';
t.querySelectorAll('th').forEach(h=>delete h.dataset.asc);th.dataset.asc=asc?'1':'0';
const val=r=>{const c=r.cells[i];return c.dataset.v!==undefined?c.dataset.v:c.textContent};
rows.sort((a,b)=>{const x=val(a),y=val(b),nx=parseFloat(x),ny=parseFloat(y);
if(!isNaN(nx)&&!isNaN(ny))return asc?nx-ny:ny-nx;if(x==='')return 1;if(y==='')return -1;
return asc?x.localeCompare(y):y.localeCompare(x)});rows.forEach(r=>tb.appendChild(r))})})});
"""


def page(title, body):
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title>"
            f"<style>{CSS}</style></head><body><main>{body}</main><script>{SORT_JS}</script></body></html>")


def e(x):
    return html.escape(str(x))


def tiles(items):
    return "<div class='tiles'>" + "".join(
        f"<div class='tile'><div class='v{' t' if len(str(v)) > 14 else ''}'>{e(v)}</div><div class='k'>{e(k)}</div></div>"
        for k, v in items) + "</div>"


def table(headers, rows, sortable=False, highlight=None):
    """rows: list of lists of (display, sort_value or None); highlight: set of row indexes."""
    cls = " class='sortable'" if sortable else ""
    h = "".join(f"<th>{e(x)}</th>" for x in headers)
    body = []
    for k, r in enumerate(rows):
        tr = " class='hl'" if highlight and k in highlight else ""
        cells = []
        for cell in r:
            disp, sv = cell if isinstance(cell, tuple) else (cell, None)
            dv = "" if sv is None else f" data-v='{sv}'"
            cells.append(f"<td{dv}>{disp}</td>")
        body.append(f"<tr{tr}>{''.join(cells)}</tr>")
    return f"<div class='wrap'><table{cls}><thead><tr>{h}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def cell(v, key):
    f = METRIC[key][3]
    return (fmt(v, f), "" if v is None else repr(v))


def _ngram_table(bl):
    if not bl:
        return ""
    splits = [s for s in ("val", "test") if s in bl]
    orders = sorted({int(k) for s in splits for k in bl[s]})
    rows = [[f"{k}-gram"] + [fmt(bl[s].get(str(k)), "{:.3f}") for s in splits] for k in orders]
    return (table(["model"] + [f"{s} BPC" for s in splits], rows) +
            f"<p class='note'>Character n-gram language models ({e(bl.get('method', ''))}) trained on the same "
            "training text. They are the classic baselines for character-level modeling: a transformer that "
            "can't beat the best n-gram isn't using its context well.</p>")


def _baselines_for(data_dir):
    try:
        from . import data as data_mod
        from . import ngram
        return ngram.baselines(data_mod.load(data_dir))
    except Exception:
        return None


# --------------------------------------------------------------- run report
def run_report(run_dir, all_runs=None):
    plt = _plt()
    run = load_run(run_dir)
    s, c = run["s"], run["cfg"]
    runs_dir = os.path.dirname(os.path.normpath(run_dir))
    all_runs = all_runs if all_runs is not None else load_runs(runs_dir)
    configs = aggregate(all_runs)
    me = next((x for x in configs if x["id"] == run["id"]), None)
    base = next((x for x in configs if x["id"] == BASELINE_ID), None)
    bl = _baselines_for(c["data_dir"])
    fe = s.get("final_eval") or {}
    v = {k: run_value(run, k) for k in METRIC}

    out = [f"<p class='muted'><a href='../../dashboard.html'>← dashboard</a></p>",
           f"<h1>{e(run['name'])}</h1>",
           "<div class='chips'>" + "".join(f"<span>{e(k)}: {e(c.get(k))}</span>" for k in
                                           ("optimizer", "activation", "norm", "positional", "lr", "seed",
                                            "n_layer", "d_model", "max_steps")) + "</div>"]
    if s.get("diverged"):
        out.append(f"<p class='warn'>This run diverged at step {s.get('steps')}. "
                   "Try a lower learning rate (scripts/lr_sweep.py --config-id …).</p>")

    out.append("<h2>Results</h2>")
    out.append(tiles([
        ("final val loss", fmt(v["val"], "{:.4f}")),
        ("bits per char", fmt(v["bpc"], "{:.3f}")),
        ("perplexity", fmt(v["ppl"], "{:.2f}")),
        ("top-1 accuracy", fmt(v["acc"], _pct)),
        ("best val (step)", f"{fmt(_get(s, 'best_val_loss'), '{:.4f}')} ({fmt(v['best_step'], '{:.0f}')})"),
        ("generalization gap", fmt(v["gap"], "{:+.4f}")),
        ("parameters", fmt(v["params"], "{:,.0f}")),
        ("chars processed", fmt(_get(s, "tokens"), "{:,.0f}")),
        ("train time", fmt(v["train_min"], "{:.1f} min")),
        ("train speed", fmt(v["tok_s"], "{:,.0f} chars/s")),
        ("peak memory", fmt(v["peak_mem"], "{:,.0f} MB")),
        ("steps clipped", fmt(v["clip"], _pct)),
    ]))
    out.append("<p class='note'>Loss is in nats per character on 20 fixed validation batches (the same for every "
               "run). BPC = loss / ln 2; perplexity = e<sup>loss</sup>. Generalization gap = validation loss − "
               "loss on fixed training batches (positive = worse on unseen text). Peak memory is the process's "
               "resident memory right after backward, when the graph is alive.</p>")

    # standing among other configs
    out.append("<h2>Compared with other runs</h2>")
    ranked = [x for x in configs if x["val"] is not None]
    lines = []
    if me and me["val"] is not None:
        pos = next(i for i, x in enumerate(ranked, 1) if x["id"] == me["id"])
        lines.append(f"This config ranks <b>{pos} of {len(ranked)}</b> configs by mean final validation loss.")
    if base and base["val"] is not None and v["val"] is not None and run["id"] != BASELINE_ID:
        d = v["val"] - base["val"]
        lines.append(f"Versus baseline 01 ({base['val']:.4f}): <b>{d:+.4f}</b> "
                     f"({'better' if d < 0 else 'worse'}).")
    if ranked and v["val"] is not None:
        lines.append(f"Best config so far: {e(ranked[0]['label'])} ({ranked[0]['val']:.4f}).")
    nb = None
    if bl and "val" in bl:
        nb = min(((int(k), x) for k, x in bl["val"].items()), key=lambda t: t[1])
        if v["bpc"] is not None:
            lines.append(f"Best n-gram baseline ({nb[0]}-gram): {nb[1]:.3f} BPC; this run: {v['bpc']:.3f} BPC "
                         f"({v['bpc'] - nb[1]:+.3f}).")
    out.append("<p>" + " ".join(lines) + "</p>")
    if plt and ranked:
        out.append(chart_rank(plt, configs, highlight=run["id"], bl=bl))

    if me and me["n_seeds"] > 1:
        out.append("<h3>All seeds of this config</h3>")
        rows = []
        hl = set()
        for k, r in enumerate(me["runs"]):
            if r["name"] == run["name"]:
                hl.add(k)
            rows.append([f"<a href='../{e(r['name'])}/report.html'>seed {r['cfg']['seed']}</a>"] +
                        [fmt(run_value(r, key), METRIC[key][3]) for key in ("val", "bpc", "acc", "full_bpc", "story")])
        rows.append(["<b>mean ± std</b>"] + [
            fmt(me[key], METRIC[key][3]) + ("" if me[key + "_std"] is None else f" ± {me[key + '_std']:.4f}")
            for key in ("val", "bpc", "acc", "full_bpc", "story")])
        out.append(table(["seed", "val loss", "BPC", "top-1 acc", "full-val BPC", "story-ending acc"], rows,
                         highlight=hl))

    out.append("<h2>Training</h2>")
    if plt and run["m"]:
        ref = []
        if base and run["id"] != BASELINE_ID:
            ref.append(("baseline 01 validation", mean_curve(base["runs"]), "--"))
        if ranked and ranked[0]["id"] not in (run["id"], BASELINE_ID):
            ref.append((f"best config {ranked[0]['id']} validation", mean_curve(ranked[0]["runs"]), ":"))
        out.append(chart_run_curves(plt, run, ref))
        out.append(chart_run_dynamics(plt, run))
        out.append(chart_run_layers(plt, run))
    ttt = s.get("time_to_target")
    if ttt:
        rows = [[f"val loss ≤ {t}", fmt(_get(x or {}, "step"), "{:.0f}", "not reached"),
                 fmt(_get(x or {}, "tokens"), "{:,.0f}", "—"), fmt(_div(_get(x or {}, "train_s"), 60), "{:.1f}", "—")]
                for t, x in sorted(ttt.items(), key=lambda kv: -float(kv[0]))]
        out.append("<h3>Time to target loss</h3>")
        out.append(table(["target", "step", "chars", "train minutes"], rows))
        out.append("<p class='note'>Measured at evaluation points (every eval_every steps), so it's a resolution "
                   "of eval_every steps. Train minutes exclude evaluation time.</p>")

    if fe:
        out.append("<h2>Evaluation</h2>")
        fv, ec, inf, ts = fe.get("full_val", {}), fe.get("ending_choice", {}), fe.get("inference", {}), fe.get(
            "text_stats", {})
        out.append(tiles([
            ("full-val BPC", fmt(_get(fv, "bpc"), "{:.3f}")),
            ("full-val top-1", fmt(_get(fv, "acc"), _pct)),
            ("chars evaluated", fmt(_get(fv, "chars"), "{:,.0f}")),
            ("story-ending acc", fmt(_get(ec, "acc"), _pct)),
            ("story-ending raw", fmt(_get(ec, "acc_raw"), _pct)),
            ("story-ending chance", fmt(_get(ec, "chance"), _pct)),
            ("generation", fmt(_get(inf, "generate_ms_per_char"), "{:.1f} ms/char")),
            ("batch throughput", fmt(_get(inf, "batch_chars_per_s"), "{:,.0f} chars/s")),
            ("real-word rate", fmt(_get(ts, "real_word"), _pct)),
            ("distinct-2", fmt(_get(ts, "distinct_2"), "{:.3f}")),
            ("rep-4", fmt(_get(ts, "rep_4"), "{:.3f}")),
        ]))
        out.append("<p class='note'><b>Full-val</b>: every character of the validation split, "
                   f"windows of {e(c.get('ctx'))} with stride {e(fv.get('stride'))}. <b>Story-ending choice</b>: "
                   f"{e(ec.get('items'))} items; the model sees a story passage and must give the real next sentence "
                   "a higher score than 3 sentences from elsewhere. The main score is calibrated: how much the "
                   "context raises the sentence's likelihood (raw likelihood favours generic sentences regardless of "
                   "context; shown as 'raw'). <b>Real-word rate</b>: share of generated "
                   "words that exist in the training text. <b>Distinct-2</b>: unique word pairs / all word pairs "
                   "(diversity). <b>Rep-4</b>: share of word 4-grams that repeat earlier in the same sample "
                   "(looping).</p>")
        out.append("<h3>Generation samples</h3><p class='note'>Fixed prompts and seeds (temperature "
                   f"{e(c.get('gen_temperature', 0.8))}, top-k {e(c.get('gen_top_k', 20))}), identical for every "
                   "run, so samples are directly comparable.</p>")
        for smp in fe.get("samples", []):
            out.append(f"<pre><b>{e(smp['prompt'])}</b>{e(smp['text'])}</pre>")

    out.append("<h2>Reference baselines</h2>")
    out.append(_ngram_table(bl))
    out.append("<details><summary>Full config</summary><pre>" + e(json.dumps(c, indent=2)) + "</pre></details>")
    out.append(f"<p class='note'>Report generated {time.strftime('%Y-%m-%d %H:%M')}. Rankings reflect the runs "
               "finished at that time; the dashboard is always current.</p>")
    path = os.path.join(run_dir, "report.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(page(run["name"], "".join(out)))
    return path


# --------------------------------------------------------------- dashboard
def _target_column(configs):
    """Lowest target loss reached by at least half the configs."""
    targets = set()
    for c in configs:
        for r in c["runs"]:
            targets.update(float(t) for t in (r["s"].get("time_to_target") or {}))
    for t in sorted(targets):
        reached = sum(any((r["s"].get("time_to_target") or {}).get(str(t)) for r in c["runs"]) for c in configs)
        if reached >= max(1, len(configs) / 2):
            return t
    return None


def _ttt(c, t):
    vals = []
    for r in c["runs"]:
        x = (r["s"].get("time_to_target") or {}).get(str(t))
        if x and x.get("train_s") is not None:
            vals.append(x["train_s"] / 60)
    return st.mean(vals) if vals else None


def build_dashboard(runs_dir, out_html, charts_dir=None):
    plt = _plt()
    runs = load_runs(runs_dir)
    if not runs:
        return None
    configs = aggregate(runs)
    bl = _baselines_for(runs[0]["cfg"]["data_dir"])
    root = os.path.dirname(os.path.abspath(out_html))
    sp = (lambda n: os.path.join(charts_dir, n)) if charts_dir else (lambda n: None)

    n_div = sum(bool(r["s"].get("diverged")) for r in runs)
    out = ["<h1>Experiment dashboard</h1>",
           f"<p class='muted'>{len(runs)} runs · {len(configs)} configs · {n_div} diverged · "
           f"data: {e(runs[0]['cfg']['data_dir'])} · updated {time.strftime('%Y-%m-%d %H:%M')}</p>"]
    ranked = [c for c in configs if c["val"] is not None]
    if ranked:
        b = ranked[0]
        base = next((c for c in configs if c["id"] == BASELINE_ID), None)
        items = [("best config", b["label"]), ("best val loss", f"{b['val']:.4f}"),
                 ("best BPC", fmt(b["bpc"], "{:.3f}")), ("seeds behind it", str(b["n_seeds"]))]
        if base and base["val"] is not None:
            items.append(("baseline 01 val loss", f"{base['val']:.4f}"))
        nb = None
        if bl and "val" in bl:
            nb = min(((int(k), x) for k, x in bl["val"].items()), key=lambda t: t[1])
            items.append((f"best n-gram ({nb[0]}-gram) BPC", f"{nb[1]:.3f}"))
        out.append(tiles(items))
        if b["n_seeds"] < 3:
            out.append(f"<p class='warn'>The leading config has only {b['n_seeds']} seed(s). Confirm the top "
                       "configs with 3 seeds before calling a winner (see the command printed by "
                       "scripts/analyze.py).</p>")

    # leaderboard
    t = _target_column(configs)
    keys = ["val", "bpc", "acc", "full_bpc", "gap", "best_step", "story", "real_word", "distinct_2", "rep_4",
            "tok_s", "train_min", "peak_mem"]
    headers = ["config", "seeds"] + [METRIC[k][1] for k in keys] + ([f"min → loss {t}"] if t else [])
    rows = []
    for c in configs:
        link = f"<a href='{e(os.path.relpath(c['runs'][0]['dir'], root)).replace(os.sep, '/')}/report.html'>{e(c['label'])}</a>"
        n = f"{c['n_seeds']}" + (f" ({c['n_diverged']} div.)" if c["n_diverged"] else "")
        row = [(link, c["id"]), (n, c["n_seeds"])]
        for k in keys:
            disp = fmt(c[k], METRIC[k][3])
            if k == "val" and c["val_std"] is not None:
                disp += f" ± {c['val_std']:.4f}"
            row.append((disp, "" if c[k] is None else repr(c[k])))
        if t:
            x = _ttt(c, t)
            row.append((fmt(x, "{:.1f}"), "" if x is None else repr(x)))
        rows.append(row)
    out.append("<h2>Leaderboard</h2>")
    out.append("<p class='note'>One row per config, mean over seeds. Click a column to sort; click a config for "
               "its full report. Primary metric: final validation loss on the fixed validation batches (decided "
               "before the experiments). Diverged runs are excluded from means.</p>")
    out.append(table(headers, rows, sortable=True))

    if plt:
        out.append("<h2>Charts</h2>")
        out.append(chart_rank(plt, configs, bl=bl, max_rows=70, save_path=sp("ranking_all.png")))
        out.append(chart_top_curves(plt, configs, save_path=sp("top_curves.png")))
        out.append(chart_scatter(plt, configs, "tok_s", "training speed (chars/s)",
                                 "Speed vs quality", sp("speed_vs_quality.png")))
        out.append(chart_scatter(plt, configs, "gap", "generalization gap (val − train loss)",
                                 "Generalization gap vs quality", sp("gap_vs_quality.png")))
        out.append("<h2>Stage 1: one change at a time</h2>")
        out.append(chart_stage1(plt, configs, sp("stage1_comparison.png")))
        out.append(chart_stage1_curves(plt, configs, sp("curves_{}.png") if charts_dir else None))
        out.append(chart_activation_telemetry(plt, configs, sp("telemetry_activations.png")))
        out.append(chart_update_ratio_by_optimizer(plt, configs, sp("telemetry_update_ratio.png")))

    confirmed = [c for c in configs if c["n_seeds"] >= 3]
    if confirmed:
        out.append("<h2>Confirmed configs (3+ seeds)</h2>")
        rows = []
        for c in confirmed:
            per = ", ".join(fmt(run_value(r, "val"), "{:.4f}", "div.") for r in c["runs"])
            rows.append([e(c["label"]), str(c["n_seeds"]),
                         fmt(c["val"], "{:.4f}") + ("" if c["val_std"] is None else f" ± {c['val_std']:.4f}"),
                         fmt(c["bpc"], "{:.3f}"), fmt(c["full_bpc"], "{:.3f}"), fmt(c["story"], _pct), per])
        out.append(table(["config", "seeds", "val loss (mean ± std)", "BPC", "full-val BPC", "story-ending acc",
                          "per seed"], rows))
        out.append("<p class='note'>Two configs are only meaningfully different if the gap between their means is "
                   "clearly larger than their standard deviations.</p>")

    ratings = _json(os.path.join(root, "human_ratings.json"))
    if ratings:
        out.append("<h2>Human ratings</h2>")
        by = {}
        for x in ratings:
            by.setdefault(x["run"], []).append(x)
        rows = [[e(k), str(len(v)), f"{st.mean(x['fluency'] for x in v):.2f}",
                 f"{st.mean(x['coherence'] for x in v):.2f}"] for k, v in sorted(by.items())]
        out.append(table(["run", "ratings", "fluency (1–5)", "coherence (1–5)"], rows, sortable=True))
        out.append("<p class='note'>Blind ratings from scripts/rate_samples.py.</p>")

    finals = [(r, _json(os.path.join(r["dir"], "final_eval.json"))) for r in runs]
    finals = [(r, f) for r, f in finals if f]
    if finals:
        out.append("<h2>Final evaluations (held-out test set)</h2>")
        rows = [[e(r["name"]), fmt(_get(f, "test", "bpc"), "{:.3f}"), fmt(_get(f, "test", "acc"), _pct),
                 fmt(_get(f, "ending_choice_test", "acc"), _pct), fmt(_get(f, "memorization", "frac_64"), _pct)]
                for r, f in finals]
        out.append(table(["run", "test BPC", "test top-1", "story-ending (test)", "generated 64-char windows in train"],
                         rows))
        out.append("<p class='note'>Full details in results/final/final_report.html.</p>")

    out.append("<h2>Reference baselines</h2>")
    out.append(_ngram_table(bl))
    os.makedirs(root, exist_ok=True)
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page("Experiment dashboard", "".join(out)))
    return out_html
