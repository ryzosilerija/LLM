"""Extra benchmarks for trained models (validation data only, never test).

  python scripts/extra_benchmarks.py results/runs          # every run in the folder (resumable)
  python scripts/extra_benchmarks.py results/runs/45_*     # some runs

Per run, writes <run>/extra_benchmarks.json and <run>/extra_benchmarks.html:

  1. Loss vs context position   how much the model gains from seeing more previous characters
  2. Length generalization      validation loss at contexts 1x, 2x, 4x the training length
                                (learned absolute positions cannot go beyond 1x)
  3. Calibration                when the model is p% sure, is it right p% of the time?
                                expected calibration error (ECE) and a reliability table
  4. Loss by character type     lowercase, uppercase, space, newline, punctuation, digits;
                                and the first letter of a word vs letters inside a word
  5. Typo robustness            loss when 5% / 10% / 20% of input characters are replaced
                                at random (targets unchanged)
  6. Sampling settings          diversity, looping and real-word rate of generated text for
                                several temperatures and top-k values
  7. Training progression       validation loss, story-ending accuracy and samples at every
                                saved checkpoint (needs checkpoint_steps during training)

All measurements use fixed random seeds, so different models are scored on
exactly the same text.
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import data as D  # noqa: E402
from nllm import metrics as M  # noqa: E402
from nllm import report as R  # noqa: E402
from nllm.backend import to_numpy, xp  # noqa: E402
from nllm.tensor import no_grad  # noqa: E402
from nllm.train import load_model  # noqa: E402

SEED = 777
VERSION = 1


def val_windows(data, ctx, n, seed=SEED):
    src = data.split_array("val")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, len(src) - ctx - 1, size=n)
    x = np.stack([src[s:s + ctx] for s in starts]).astype(np.int64)
    y = np.stack([src[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
    return x, y


def forward_stats(model, x, y, batch=64):
    """Per-position NLL, correctness and top-1 probability for windows (x, y)."""
    nll_all, cor_all, conf_all = [], [], []
    with no_grad():
        for b in range(0, len(x), batch):
            xb, yb = xp.asarray(x[b:b + batch]), xp.asarray(y[b:b + batch])
            logits = model(xb).data
            m = logits.max(-1, keepdims=True)
            lse = m[..., 0] + xp.log(xp.exp(logits - m).sum(-1))
            tgt = xp.take_along_axis(logits, yb[..., None], -1)[..., 0]
            nll_all.append(to_numpy(lse - tgt).astype(np.float64))
            cor_all.append(to_numpy(logits.argmax(-1) == yb))
            conf_all.append(to_numpy(xp.exp(m[..., 0] - lse)).astype(np.float64))
    return np.concatenate(nll_all), np.concatenate(cor_all), np.concatenate(conf_all)


def context_position(nll):
    return [float(v) for v in nll.mean(0)]


def calibration(cor, conf, bins=10):
    c, k = conf.reshape(-1), cor.reshape(-1).astype(np.float64)
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        sel = (c > edges[i]) & (c <= edges[i + 1]) if i else (c >= 0) & (c <= edges[1])
        n = int(sel.sum())
        if n:
            acc, avg = float(k[sel].mean()), float(c[sel].mean())
            ece += n / len(c) * abs(acc - avg)
            rows.append({"bin": f"{edges[i]:.1f}-{edges[i + 1]:.1f}", "n": n, "confidence": avg, "accuracy": acc})
    return {"ece": ece, "mean_confidence": float(c.mean()), "accuracy": float(k.mean()), "bins": rows}


def char_classes(data, x, y, nll):
    chars = data.chars
    kind = np.array([("lowercase" if ch.islower() else "uppercase" if ch.isupper() else "digit" if ch.isdigit()
                      else "space" if ch == " " else "newline" if ch == "\n" else "punctuation")
                     for ch in chars])
    t = kind[y]
    out = {}
    for name in ("lowercase", "uppercase", "space", "newline", "punctuation", "digit"):
        sel = t == name
        if sel.any():
            out[name] = {"loss": float(nll[sel].mean()), "share": float(sel.mean())}
    # first letter of a word (previous char is a space/newline) vs inside a word
    is_letter = np.array([ch.isalpha() for ch in chars])
    prev_is_space = np.isin(x, [i for i, ch in enumerate(chars) if ch in " \n"])
    letter = is_letter[y]
    out["word start"] = {"loss": float(nll[letter & prev_is_space].mean()), "share": float((letter & prev_is_space).mean())}
    inside = letter & ~prev_is_space
    out["inside word"] = {"loss": float(nll[inside].mean()), "share": float(inside.mean())}
    return out


def robustness(model, data, x, y, rates=(0.0, 0.05, 0.10, 0.20)):
    rng = np.random.default_rng(SEED + 1)
    out = {}
    for r in rates:
        xc = x.copy()
        if r:
            mask = rng.random(x.shape) < r
            xc[mask] = rng.integers(0, data.vocab_size, size=int(mask.sum()))
        nll, _, _ = forward_stats(model, xc, y)
        out[f"{r:.2f}"] = float(nll.mean())
    return out


def length_gen(model, data, cfg, factors=(1, 2, 4), n_chars=262144):
    out = {}
    for f in factors:
        L = cfg["ctx"] * f
        if cfg["positional"] == "learned" and L > cfg["ctx"]:
            out[str(L)] = None
            continue
        x, y = val_windows(data, L, max(4, n_chars // L), seed=SEED + 2)
        nll, _, _ = forward_stats(model, x, y, batch=max(1, 64 // f))
        out[str(L)] = {"loss_all": float(nll.mean()), "loss_last_quarter": float(nll[:, -L // 4:].mean())}
    return out


SAMPLING = [(0.5, 20), (0.8, 20), (1.0, 20), (0.8, 0), (1.0, 0)]


def sampling_sweep(model, data, cfg, chars=200):
    prompts = (cfg.get("gen_prompts") or M.DEFAULT_PROMPTS)[:3]
    out = []
    for temp, k in SAMPLING:
        texts = []
        for i, p in enumerate(prompts):
            ids, _ = M.generate(model, cfg["ctx"], data.encode(p), chars, temp, k, seed=100 + i)
            texts.append(data.decode(ids))
        st = M.text_stats(texts, data.train_words())
        out.append({"temperature": temp, "top_k": k or "none", **{k2: st[k2] for k2 in ("distinct_2", "rep_4", "real_word")},
                    "example": prompts[0] + texts[0]})
    return out


def progression(run_dir, data, cfg, items):
    pts = []
    ck_rows = {}
    path = os.path.join(run_dir, "checkpoints.jsonl")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            ck_rows[r["step"]] = r
    files = sorted(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.npz")))
    final = os.path.join(run_dir, "model.npz")
    for f in files + ([final] if os.path.exists(final) else []):
        model, _, _ = load_model(run_dir, f)
        step = int(os.path.basename(f)[5:11]) if f != final else cfg["max_steps"]
        v = M.eval_batches(model, data, "val", cfg["batch_size"], cfg["ctx"], cfg["eval_batches"], 12345)
        ec = M.ending_choice(model, items, cfg["ctx"], data.stoi.get("\n"))
        pts.append({"step": step, "tokens": step * cfg["batch_size"] * cfg["ctx"], "val_loss": v["loss"],
                    "val_bpc": v["bpc"], "story": ec["acc"], "story_raw": ec["acc_raw"],
                    "samples": (ck_rows.get(step) or {}).get("samples", [])})
    return pts


# ------------------------------------------------------------------ html
def html_report(run_dir, res):
    plt = R._plt()
    e, f = R.e, R.fmt
    name = os.path.basename(os.path.normpath(run_dir))
    out = [f"<p class='muted'><a href='report.html'>← run report</a></p><h1>Extra benchmarks: {e(name)}</h1>",
           "<p class='note'>All measured on validation text with fixed seeds (the test set is untouched). "
           "Losses are in nats per character; BPC = loss / ln 2.</p>"]

    out.append("<h2>1. Loss vs context position</h2>")
    cp = res["context_position"]
    if plt:
        fig, ax = plt.subplots(figsize=(8.5, 3.8))
        R._style(ax)
        ax.plot(range(len(cp)), cp, color=R.PALETTE[0], lw=2)
        ax.set_xlabel("position in the context window (characters of history available)")
        ax.set_ylabel("loss")
        ax.set_title("How much the model gains from more context", loc="left", fontsize=11)
        out.append(R._embed(fig, plt, "loss by context position"))
    pick = [0, 1, 2, 4, 8, 16, 32, 64, len(cp) - 1]
    out.append(R.table(["position"] + [str(p) for p in pick if p < len(cp)],
                       [["loss"] + [f"{cp[p]:.3f}" for p in pick if p < len(cp)]]))
    out.append("<p class='note'>Position 0 has no history, so it's a pure guess from character frequencies. The "
               "curve shows how many characters of context the model actually uses before it flattens.</p>")

    out.append("<h2>2. Length generalization</h2>")
    rows = [[L, "n/a (learned positions)" if v is None else f"{v['loss_all']:.4f}",
             "—" if v is None else f"{v['loss_last_quarter']:.4f}"] for L, v in res["length_generalization"].items()]
    out.append(R.table(["context length", "loss (all positions)", "loss (last quarter of window)"], rows))
    out.append("<p class='note'>The model was trained with the first length. Longer contexts test whether it can use "
               "positions it never saw; 'last quarter' isolates the positions beyond training length.</p>")

    out.append("<h2>3. Calibration</h2>")
    cal = res["calibration"]
    out.append(R.tiles([("ECE", f"{cal['ece']:.3f}"), ("mean confidence", f"{cal['mean_confidence']:.1%}"),
                        ("accuracy", f"{cal['accuracy']:.1%}")]))
    if plt:
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        R._style(ax, "both")
        ax.plot([0, 1], [0, 1], color=R.INK2, ls="--", lw=1)
        ax.plot([b["confidence"] for b in cal["bins"]], [b["accuracy"] for b in cal["bins"]], color=R.PALETTE[0],
                lw=2, marker="o", ms=5)
        ax.set_xlabel("confidence of top prediction")
        ax.set_ylabel("actual accuracy")
        ax.set_title("Reliability diagram", loc="left", fontsize=11)
        out.append(R._embed(fig, plt, "reliability diagram"))
    out.append(R.table(["confidence bin", "predictions", "mean confidence", "accuracy"],
                       [[b["bin"], f"{b['n']:,}", f"{b['confidence']:.1%}", f"{b['accuracy']:.1%}"] for b in cal["bins"]]))
    out.append("<p class='note'>ECE = average gap between confidence and accuracy, weighted by how many predictions "
               "fall in each bin (0 = perfectly calibrated). On the diagonal = calibrated; below = overconfident.</p>")

    out.append("<h2>4. Loss by character type</h2>")
    out.append(R.table(["target character", "share of text", "loss", "BPC"],
                       [[k, f"{v['share']:.1%}", f"{v['loss']:.3f}", f"{v['loss'] / math.log(2):.3f}"]
                        for k, v in res["char_classes"].items()]))
    out.append("<p class='note'>Word starts are where the model has to decide what comes next; letters inside a word "
               "are mostly determined by the word so far.</p>")

    out.append("<h2>5. Typo robustness</h2>")
    rb = res["robustness"]
    base = rb.get("0.00")
    out.append(R.table(["input characters corrupted", "loss", "increase"],
                       [[f"{float(k):.0%}", f"{v:.4f}", "—" if k == "0.00" else f"{v - base:+.4f}"] for k, v in rb.items()]))

    out.append("<h2>6. Sampling settings</h2>")
    out.append(R.table(["temperature", "top-k", "distinct-2", "rep-4", "real words"],
                       [[s["temperature"], s["top_k"], f"{s['distinct_2']:.3f}", f"{s['rep_4']:.3f}",
                         f"{s['real_word']:.1%}"] for s in res["sampling"]]))
    for s in res["sampling"]:
        out.append(f"<pre><b>T={s['temperature']}, top-k={s['top_k']}</b>\n{e(s['example'])}</pre>")

    prog = res.get("progression") or []
    if prog:
        out.append("<h2>7. Training progression</h2>")
        if plt and len(prog) > 1:
            fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.6))
            for ax in (a1, a2):
                R._style(ax)
                ax.set_xscale("log")
                ax.set_xlabel("training characters")
            a1.plot([p["tokens"] for p in prog], [p["val_bpc"] for p in prog], color=R.PALETTE[0], lw=2, marker="o")
            a1.set_title("Validation BPC", loc="left", fontsize=10)
            a2.plot([p["tokens"] for p in prog], [p["story"] for p in prog], color=R.PALETTE[1], lw=2, marker="o")
            a2.axhline(0.25, color=R.INK2, ls="--", lw=1)
            a2.set_title("Story-ending accuracy (chance 25%)", loc="left", fontsize=10)
            a2.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
            fig.tight_layout()
            out.append(R._embed(fig, plt, "training progression"))
        out.append(R.table(["step", "chars seen", "val BPC", "story-ending cal / raw"],
                           [[p["step"], f"{p['tokens']:,}", f"{p['val_bpc']:.3f}",
                             f"{p['story']:.1%} / {p['story_raw']:.1%}"] for p in prog]))
        out.append("<h3>How the generated text develops</h3>")
        for p in prog:
            for smp in p["samples"][:1]:
                out.append(f"<pre><b>step {p['step']:,}: {e(smp['prompt'])}</b>{e(smp['text'])}</pre>")
    with open(os.path.join(run_dir, "extra_benchmarks.html"), "w", encoding="utf-8") as fh:
        fh.write(R.page(f"Extra benchmarks {name}", "".join(out)))


def expand(patterns):
    """Accept run dirs, globs (PowerShell doesn't expand *) or a runs folder."""
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p)) or [p]
        for h in hits:
            if os.path.exists(os.path.join(h, "summary.json")):
                out.append(h)
            else:  # a folder of runs
                out += sorted(os.path.dirname(s) for s in glob.glob(os.path.join(h, "*", "summary.json")))
    return list(dict.fromkeys(out))


def benchmark_run(run_dir, args):
    t0 = time.time()
    model, cfg, _ = load_model(run_dir)
    data = D.load(cfg["data_dir"])
    x, y = val_windows(data, cfg["ctx"], args.windows)
    nll, cor, conf = forward_stats(model, x, y)
    res = {"version": VERSION, "run": os.path.basename(os.path.normpath(run_dir)), "windows": args.windows,
           "context_position": context_position(nll),
           "length_generalization": length_gen(model, data, cfg),
           "calibration": calibration(cor, conf),
           "char_classes": char_classes(data, x, y, nll),
           "robustness": robustness(model, data, x[:512], y[:512]),
           "sampling": sampling_sweep(model, data, cfg)}
    has_ckpt = bool(glob.glob(os.path.join(run_dir, "checkpoints", "step_*.npz")))
    if has_ckpt:
        items = M.ending_choice_items(data, "val", cfg["ctx"], args.story_items, seed=5)
        res["progression"] = progression(run_dir, data, cfg, items)
    else:
        res["progression"] = []
    res["seconds"] = time.time() - t0
    with open(os.path.join(run_dir, "extra_benchmarks.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    html_report(run_dir, res)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run dirs, globs like results/runs/45_*, or a runs folder")
    ap.add_argument("--windows", type=int, default=2048, help="validation windows for sections 1, 3, 4, 5")
    ap.add_argument("--story-items", type=int, default=200, help="story-ending items per checkpoint")
    ap.add_argument("--force", action="store_true", help="redo runs that already have extra_benchmarks.json")
    args = ap.parse_args()
    runs = expand(args.runs)
    todo, skipped = [], 0
    for r in runs:
        s = json.load(open(os.path.join(r, "summary.json")))
        old = os.path.join(r, "extra_benchmarks.json")
        if s.get("diverged") or not os.path.exists(os.path.join(r, "model.npz")):
            skipped += 1
        elif os.path.exists(old) and not args.force and (json.load(open(old)).get("version") == VERSION):
            skipped += 1
        else:
            todo.append(r)
    print(f"{len(runs)} runs found, {skipped} skipped (done / diverged / no model), {len(todo)} to benchmark")
    t_start = time.time()
    for k, run_dir in enumerate(todo, 1):
        try:
            res = benchmark_run(run_dir, args)
        except KeyboardInterrupt:
            print("\ninterrupted; rerun the same command to resume")
            return
        lg = res["length_generalization"]
        eta = (time.time() - t_start) / k * (len(todo) - k) / 60
        print(f"[{k}/{len(todo)}] {res['run']}: ECE {res['calibration']['ece']:.3f}, "
              f"loss pos0 {res['context_position'][0]:.3f} -> last {res['context_position'][-1]:.3f}, ctx "
              + ", ".join(f"{L}: {'n/a' if v is None else round(v['loss_all'], 3)}" for L, v in lg.items())
              + (f", {len(res['progression'])} checkpoints" if res["progression"] else "")
              + f" [{res['seconds']:.0f} s, ETA {eta:.0f} min]", flush=True)


if __name__ == "__main__":
    main()
