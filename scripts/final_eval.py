"""Final evaluation on the held-out TEST split.

  python scripts/final_eval.py results/runs/21_normuon_swiglu_rmsnorm_rope_s1 [more run dirs]

Run this only on your final candidates, after you've picked them using
validation results. If you choose between configs by their test score, the
test set becomes a second validation set and stops being an honest estimate.

Per run (saved to <run>/final_eval.json):
  test loss / BPC / perplexity / top-1 accuracy over the whole test split
  (sliding windows, stride ctx/2), plus the same on the full validation split
  story-ending choice accuracy on test items
  inference latency (ms per generated char) and throughput (batch and single)
  generation samples, diversity, real-word rate
  memorization: share of generated 32/64-char windows that appear verbatim in
  the training text, and the longest verbatim match
  human ratings, if scripts/rate_samples.py was used
Dataset level (cached as leakage.json in the data dir):
  share of val/test windows (32/64/128 chars) that also appear in train
Writes results/final/final_report.html comparing all evaluated runs.
"""
import argparse
import json
import os
import statistics as st
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import data as D  # noqa: E402
from nllm import metrics as M  # noqa: E402
from nllm import ngram  # noqa: E402
from nllm import report as R  # noqa: E402
from nllm.train import load_model  # noqa: E402

LEAK_LENGTHS = (32, 64, 128)
MATCH_LENGTHS = (16, 32, 64, 128)


class TrainIndex:
    """Sorted unique hashes of every training window, per length (lazy)."""

    def __init__(self, data):
        self.data, self.cache = data, {}

    def __getitem__(self, L):
        if L not in self.cache:
            self.cache[L] = np.unique(M.window_hashes(self.data.train, L))
        return self.cache[L]


def dataset_leakage(data, index):
    path = os.path.join(data.data_dir, "leakage.json")
    cached = R._json(path)
    if cached and cached.get("train_chars") == int(len(data.train)):
        return cached
    out = {"train_chars": int(len(data.train))}
    for split in ("val", "test"):
        if data.has(split):
            out[split] = {str(L): M.window_overlap(data.split_array(split), index[L], L) for L in LEAK_LENGTHS}
    json.dump(out, open(path, "w"), indent=2)
    return out


def evaluate_run(run_dir, data, index, args, ratings):
    model, cfg, chars = load_model(run_dir)
    ctx = cfg["ctx"]
    stride = args.stride or max(1, ctx // 2)
    out = {"run": os.path.basename(os.path.normpath(run_dir)), "evaluated": time.strftime("%Y-%m-%d %H:%M")}
    t0 = time.time()
    out["test"] = M.full_split_eval(model, data, "test", ctx, stride)
    out["val_full"] = M.full_split_eval(model, data, "val", ctx, stride)
    items = M.ending_choice_items(data, "test", ctx, args.choice_items, seed=1)
    out["ending_choice_test"] = M.ending_choice(model, items, ctx, data.stoi.get("\n"))

    prompts = cfg.get("gen_prompts") or M.DEFAULT_PROMPTS
    samples, gen_ids, gen_time, n_chars = [], [], 0.0, 0
    for k in range(args.samples_per_prompt):
        for i, p in enumerate(prompts):
            ids, dt = M.generate(model, ctx, data.encode(p), args.chars, cfg.get("gen_temperature", 0.8),
                                 cfg.get("gen_top_k", 20), seed=500 + 100 * k + i)
            samples.append({"prompt": p, "text": data.decode(ids)})
            gen_ids.append(np.asarray(ids, dtype=np.uint16))
            gen_time += dt
            n_chars += len(ids)
    out["samples"] = samples
    out["text_stats"] = M.text_stats([s["text"] for s in samples], data.train_words())
    out["memorization"] = {
        "frac_32": float(np.mean([M.window_overlap(g, index[32], 32) for g in gen_ids])),
        "frac_64": float(np.mean([M.window_overlap(g, index[64], 64) for g in gen_ids])),
        "longest_match_max": int(max(M.longest_match(g, {L: index[L] for L in MATCH_LENGTHS}) for g in gen_ids)),
    }
    speed = M.inference_speed(model, ctx, data.vocab_size)
    speed["generate_ms_per_char"] = 1000 * gen_time / max(n_chars, 1)
    out["inference"] = speed
    mine = [r for r in ratings if r.get("run") == out["run"]]
    if mine:
        out["human"] = {"n": len(mine), "fluency": st.mean(r["fluency"] for r in mine),
                        "coherence": st.mean(r["coherence"] for r in mine)}
    out["n_params"] = int(model.n_params())
    out["config"] = {k: cfg[k] for k in ("optimizer", "activation", "norm", "positional", "lr", "seed", "n_layer",
                                         "d_model", "n_head", "ctx", "max_steps")}
    out["seconds"] = time.time() - t0
    json.dump(out, open(os.path.join(run_dir, "final_eval.json"), "w"), indent=2)
    return out


def final_report(results, data, bl, leak, path):
    plt = R._plt()
    f, e, pct = R.fmt, R.e, R._pct
    body = ["<h1>Final evaluation (held-out test set)</h1>",
            f"<p class='muted'>{len(results)} model(s) · data: {e(data.data_dir)} · test split: "
            f"{len(data.split_array('test')):,} characters · {time.strftime('%Y-%m-%d %H:%M')}</p>",
            "<p class='warn'>These numbers are only honest if the models were chosen using validation results. "
            "Don't go back and pick a different config because of its test score.</p>"]
    rows = []
    for r in results:
        h = r.get("human") or {}
        rows.append([e(r["run"]), f(R._get(r, "n_params"), "{:,.0f}"),
                     f(R._get(r, "test", "loss"), "{:.4f}"), f(R._get(r, "test", "bpc"), "{:.3f}"),
                     f(R._get(r, "test", "ppl"), "{:.2f}"), f(R._get(r, "test", "acc"), pct),
                     f(R._get(r, "val_full", "bpc"), "{:.3f}"),
                     f(R._get(r, "ending_choice_test", "acc"), pct), f(R._get(r, "ending_choice_test", "acc_raw"), pct),
                     f(R._get(r, "text_stats", "real_word"), pct), f(R._get(r, "text_stats", "distinct_2"), "{:.3f}"),
                     f(R._get(r, "text_stats", "rep_4"), "{:.3f}"),
                     f(R._get(h, "fluency"), "{:.2f}"), f(R._get(h, "coherence"), "{:.2f}"),
                     f(R._get(r, "inference", "generate_ms_per_char"), "{:.1f}"),
                     f(R._get(r, "inference", "batch_chars_per_s"), "{:,.0f}")])
    body.append("<h2>Results</h2>")
    body.append(R.table(["run", "params", "test loss", "test BPC", "test ppl", "test top-1", "full-val BPC",
                         "story-ending", "story-ending raw", "real words", "distinct-2", "rep-4", "fluency", "coherence",
                         "ms / generated char", "batch chars/s"], rows, sortable=True))
    body.append("<p class='note'>Test and full-val metrics score every character of the split with sliding "
                "windows (each character sees at least half a context of history). A test BPC much worse than "
                "full-val BPC suggests the choice of config overfit the validation set. Story-ending chance level "
                "is 25%. Fluency and coherence are blind human ratings (1–5) from scripts/rate_samples.py.</p>")

    if plt:
        fig, ax = plt.subplots(figsize=(9, 0.5 * len(results) + 1.6))
        R._style(ax, "x")
        names = [r["run"] for r in results]
        y = np.arange(len(results))
        ax.barh(y - 0.18, [R._get(r, "val_full", "bpc") or 0 for r in results], height=0.34,
                color=R.PALETTE[1], label="full validation split")
        ax.barh(y + 0.18, [R._get(r, "test", "bpc") or 0 for r in results], height=0.34,
                color=R.PALETTE[0], label="test split")
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        if bl and "test" in bl:
            o, v = ngram.best(bl, "test")
            ax.axvline(v, color=R.INK2, ls="--", lw=1)
            ax.text(v, 1.0, f" best n-gram on test ({o}-gram) {v:.2f}", transform=ax.get_xaxis_transform(),
                    fontsize=8, color=R.INK2, va="bottom")
        vals = [x for r in results for x in (R._get(r, "test", "bpc"), R._get(r, "val_full", "bpc")) if x]
        if bl and "test" in bl:
            vals.append(ngram.best(bl, "test")[1])
        ax.set_xlim(min(vals) - 0.15, max(vals) + 0.1)
        ax.set_xlabel("bits per character (lower is better)")
        ax.set_title("Test vs full-validation BPC", loc="left", fontsize=11, pad=18)
        ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        body.append(R._embed(fig, plt, "test versus validation bits per character"))

    body.append("<h2>Data leakage checks</h2>")
    rows = [[s] + [f(R._get(leak, s, str(L)), pct) for L in LEAK_LENGTHS] for s in ("val", "test") if s in leak]
    body.append(R.table(["split"] + [f"{L}-char windows also in train" for L in LEAK_LENGTHS], rows))
    rows = [[e(r["run"]), f(R._get(r, "memorization", "frac_32"), pct), f(R._get(r, "memorization", "frac_64"), pct),
             f(R._get(r, "memorization", "longest_match_max"), "{:.0f}")] for r in results]
    body.append(R.table(["run", "generated 32-char windows in train", "generated 64-char windows in train",
                         "longest verbatim match (chars, of 16/32/64/128)"], rows))
    body.append("<p class='note'>Short windows overlap naturally in children's stories (\"Once upon a time, there "
                "was a little\"). What matters is the long windows: if many 128-character test windows also occur in "
                "train, the test split has near-duplicates of training text and its score is optimistic; if "
                "generations contain long verbatim training passages, the model is copying rather than generating. "
                "Compare the generated-text rates with the test-split rates: generated text that overlaps training "
                "much more than real held-out text does is a sign of memorization.</p>")

    body.append("<h2>Generation samples</h2>")
    for r in results:
        body.append(f"<h3>{e(r['run'])}</h3>")
        for smp in r["samples"][:5]:
            body.append(f"<pre><b>{e(smp['prompt'])}</b>{e(smp['text'])}</pre>")
    body.append("<h2>Reference baselines</h2>")
    body.append(R._ngram_table(bl))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(R.page("Final evaluation", "".join(body)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default="results/final")
    ap.add_argument("--stride", type=int, default=None, help="sliding-window stride (default ctx/2)")
    ap.add_argument("--choice-items", type=int, default=1000)
    ap.add_argument("--samples-per-prompt", type=int, default=2)
    ap.add_argument("--chars", type=int, default=300)
    ap.add_argument("--ratings", default="results/human_ratings.json")
    args = ap.parse_args()

    cfg0 = json.load(open(os.path.join(args.runs[0], "config.json")))
    data = D.load(cfg0["data_dir"])
    if not data.has("test"):
        raise SystemExit(f"{cfg0['data_dir']} has no test.bin. Rerun scripts/prepare_data.py with the same arguments "
                         "as before: it rewrites identical train/val files and adds test.bin.")
    ratings = R._json(args.ratings, []) or []
    print("indexing training text for leakage checks ...")
    index = TrainIndex(data)
    leak = dataset_leakage(data, index)
    bl = ngram.baselines(data)
    results = []
    for run_dir in args.runs:
        print(f"evaluating {run_dir} ...")
        r = evaluate_run(run_dir, data, index, args, ratings)
        results.append(r)
        print(f"  test BPC {r['test']['bpc']:.3f}  (full val {r['val_full']['bpc']:.3f})  "
              f"story-ending {r['ending_choice_test']['acc']:.1%}  [{r['seconds']:.0f} s]")
    path = os.path.join(args.out, "final_report.html")
    final_report(results, data, bl, leak, path)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
