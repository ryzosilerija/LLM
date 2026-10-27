"""One training run: train(config, out_dir) -> summary dict.

Writes to out_dir:
  config.json     the exact config used
  metrics.jsonl   one JSON row per logged step
  summary.json    final results (written last; its presence = run done)
  model.npz       final weights + vocab, for generation and later evaluation
  report.html     a readable report with charts and comparisons to other runs

Logged every `log_every` steps (A):
  train loss and perplexity, learning rate, gradient norm, gradient-clipping
  rate, elapsed time, characters processed, memory; per layer group: gradient
  norm, weight norm, update ratio, Loss Change Allocation; FFN dead units and
  sparsity, attention entropy, residual-stream norms.
At every eval (`eval_every`):
  validation loss / accuracy / BPC / perplexity, the same on fixed training
  batches, and the generalization gap (val - train).

Summary (B): best validation loss and its step, BPC, accuracy, gap, parameter
count, peak memory, time to each target loss, and (if final_eval) full
validation-split metrics, generation samples, diversity, story-ending choice
accuracy and inference speed.

Nothing added here touches the training math or the random streams for
data, init or Sophia, so results match runs made before these metrics existed.
"""
import json
import math
import os
import time

import numpy as np

from . import data as data_mod
from . import metrics as M
from . import tensor as T
from .backend import to_numpy, xp
from .nn import GPT, layer_group
from .optim import build_optimizer
from .schedule import lr_multiplier

DEFAULTS = {
    "data_dir": "data/tinystories",
    "optimizer": "adamw",
    "activation": "gelu",
    "norm": "layernorm",
    "positional": "learned",
    "lr": None,
    "weight_decay": 0.1,
    "schedule": "cosine",
    "warmup": 100,
    "min_lr_ratio": 0.1,
    "grad_clip": 1.0,
    "n_layer": 2,
    "d_model": 64,
    "n_head": 4,
    "ctx": 128,
    "batch_size": 32,
    "max_steps": 2000,
    "eval_every": 200,
    "eval_batches": 20,
    "log_every": 50,
    "seed": 1,
    "diverge_factor": 3.0,
    "save_model": True,
    # metrics
    "train_eval": True,  # fixed training batches at each eval -> generalization gap
    "target_val_losses": [2.0, 1.8, 1.6, 1.4, 1.2],
    "final_eval": True,  # full val split, samples, diversity, story-ending benchmark, speed
    "full_eval_stride": None,  # None = ctx (non-overlapping windows)
    "gen_prompts": None,  # None = metrics.DEFAULT_PROMPTS
    "gen_chars": 300,
    "gen_temperature": 0.8,
    "gen_top_k": 20,
    "choice_items": 200,
    "write_report": True,
    # checkpoints: save weights + validation loss + fixed-prompt samples after these steps
    # (e.g. [500, 1000, 5000]); empty = none. Used to show how the model develops.
    "checkpoint_steps": [],
    "checkpoint_sample_chars": 200,
}

VAL_SEED, TRAIN_EVAL_SEED = 12345, 54321


def _global_norm(arrays):
    return float(math.sqrt(sum(float((a * a).sum()) for a in arrays)))


def evaluate(model, data, cfg, n_batches=None, ctx=None, split="val"):
    """Metrics on a fixed set of batches (identical for every run).
    Returns a dict: loss, acc, bpc, ppl."""
    seed = VAL_SEED if split == "val" else TRAIN_EVAL_SEED
    return M.eval_batches(model, data, split, cfg["batch_size"], ctx or cfg["ctx"],
                          n_batches or cfg["eval_batches"], seed)


def final_evaluation(model, data, cfg):
    """Post-training evaluation used in every run's report."""
    out = {}
    t0 = time.time()
    out["full_val"] = M.full_split_eval(model, data, "val", cfg["ctx"], cfg["full_eval_stride"])
    prompts = cfg["gen_prompts"] or M.DEFAULT_PROMPTS
    samples, gen_time, gen_chars = [], 0.0, 0
    for i, p in enumerate(prompts):
        ids, dt = M.generate(model, cfg["ctx"], data.encode(p), cfg["gen_chars"], cfg["gen_temperature"],
                             cfg["gen_top_k"], seed=i)
        samples.append({"prompt": p, "text": data.decode(ids)})
        gen_time += dt
        gen_chars += len(ids)
    out["samples"] = samples
    out["text_stats"] = M.text_stats([s["text"] for s in samples], data.train_words())
    items = M.ending_choice_items(data, "val", cfg["ctx"], cfg["choice_items"])
    out["ending_choice"] = M.ending_choice(model, items, cfg["ctx"], data.stoi.get("\n"))
    speed = M.inference_speed(model, cfg["ctx"], data.vocab_size)
    speed["generate_ms_per_char"] = 1000 * gen_time / max(gen_chars, 1)
    out["inference"] = speed
    out["seconds"] = time.time() - t0
    return out


def _forward_loss(model, x, y):
    logits = model(x)
    V = logits.shape[-1]
    return T.cross_entropy(logits.reshape(-1, V), y.reshape(-1)), logits


def train(cfg, out_dir=None, verbose=True):
    cfg = {**DEFAULTS, **cfg}
    data = data_mod.load(cfg["data_dir"])
    model = GPT(cfg, data.vocab_size, seed=cfg["seed"])
    named = model.named_parameters()
    params = dict(named)
    groups = {n: layer_group(n) for n, _ in named}
    group_names = sorted(set(groups.values()))
    opt = build_optimizer(cfg["optimizer"], named, cfg["lr"], cfg["weight_decay"])
    rng_data = np.random.default_rng(cfg["seed"] + 1000)  # same batches for every config with this seed
    rng_hess = np.random.default_rng(cfg["seed"] + 2000)

    log_f = None
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
        log_f = open(os.path.join(out_dir, "metrics.jsonl"), "w")
        ck_log = os.path.join(out_dir, "checkpoints.jsonl")
        if os.path.exists(ck_log):
            os.remove(ck_log)

    lca = {g: 0.0 for g in group_names}
    init_loss = math.log(data.vocab_size)
    train_ema, best_val, best_val_step, diverged, step = None, float("inf"), None, False, 0
    evals = []
    tokens_per_step = cfg["batch_size"] * cfg["ctx"]
    targets = sorted(cfg["target_val_losses"] or [], reverse=True)
    time_to_target = {str(t): None for t in targets}
    clipped_total, clipped_window, window_steps = 0, 0, 0
    peak_mem = M.memory_mb() or 0.0
    eval_time = 0.0
    last_val = last_train_eval = None
    ckpt_steps = {int(x) for x in (cfg.get("checkpoint_steps") or []) if 0 < int(x) < cfg["max_steps"]}
    t0 = time.time()

    for step in range(cfg["max_steps"]):
        opt.lr = cfg["lr"] * lr_multiplier(step, cfg["max_steps"], cfg["warmup"], cfg["schedule"],
                                           cfg["min_lr_ratio"])
        x, y = data.batch("train", cfg["batch_size"], cfg["ctx"], rng_data)

        # Sophia: refresh the diagonal Hessian estimate with sampled labels
        if opt.needs_hessian and step % opt.hessian_every == 0:
            model.zero_grad()
            logits = model(x)
            V = logits.shape[-1]
            p = to_numpy(T.softmax(logits, -1).data).reshape(-1, V).astype(np.float64)
            p /= p.sum(1, keepdims=True)
            u = rng_hess.random((p.shape[0], 1))
            ys = np.minimum((p.cumsum(1) < u).sum(1), V - 1)
            T.cross_entropy(logits.reshape(-1, V), xp.asarray(ys)).backward()
            opt.update_hessian({n: q.grad for n, q in named}, tokens_per_step)

        record = step % cfg["log_every"] == 0 or step == cfg["max_steps"] - 1
        model.set_record(record)
        model.zero_grad()
        loss, _ = _forward_loss(model, x, y)
        loss.backward()
        loss_v = loss.item()
        mem = M.memory_mb()  # graph and gradients are alive here: close to the step's peak
        if mem:
            peak_mem = max(peak_mem, mem)
        if record:
            fwd_stats = model.telemetry()
            model.set_record(False)  # keep eval forwards from overwriting stats

        if not math.isfinite(loss_v) or loss_v > cfg["diverge_factor"] * init_loss:
            diverged = True
            if verbose:
                print(f"  diverged at step {step} (loss={loss_v})")
            break

        grads = [q.grad for _, q in named]
        gnorm = _global_norm(grads)
        clipped = bool(cfg["grad_clip"] and gnorm > cfg["grad_clip"])
        if clipped:
            scale = cfg["grad_clip"] / (gnorm + 1e-12)
            for _, q in named:
                q.grad = q.grad * scale
        clipped_total += clipped
        clipped_window += clipped
        window_steps += 1

        old = {n: q.data.copy() for n, q in named}
        opt.step()

        # first-order loss change allocation: sum of g * dtheta per group
        for n, q in named:
            lca[groups[n]] += float((q.grad * (q.data - old[n])).sum())

        train_ema = loss_v if train_ema is None else 0.98 * train_ema + 0.02 * loss_v

        if record:
            elapsed = time.time() - t0
            row = {"step": step, "loss": loss_v, "train_ema": train_ema, "ppl": M.ppl(train_ema), "lr": opt.lr,
                   "grad_norm": gnorm, "clip_rate": clipped_window / max(window_steps, 1),
                   "clip_rate_total": clipped_total / (step + 1), "elapsed_s": elapsed,
                   "train_s": elapsed - eval_time, "tokens": (step + 1) * tokens_per_step, "mem_mb": mem,
                   "groups": {}}
            clipped_window = window_steps = 0
            for g in group_names:
                names = [n for n in groups if groups[n] == g]
                w = _global_norm([params[n].data for n in names])
                d = _global_norm([params[n].data - old[n] for n in names])
                row["groups"][g] = {
                    "grad_norm": _global_norm([params[n].grad for n in names]),
                    "weight_norm": w,
                    "update_ratio": d / (w + 1e-12),
                    "lca": lca[g],
                }
            row["telemetry"] = fwd_stats
            if step % cfg["eval_every"] == 0 or step == cfg["max_steps"] - 1:
                te = time.time()
                v = evaluate(model, data, cfg)
                row.update(val_loss=v["loss"], val_acc=v["acc"], val_bpc=v["bpc"], val_ppl=v["ppl"])
                last_val = v
                if cfg["train_eval"]:
                    tr = evaluate(model, data, cfg, split="train")
                    row.update(train_eval_loss=tr["loss"], train_eval_acc=tr["acc"], gap=v["loss"] - tr["loss"])
                    last_train_eval = tr
                eval_time += time.time() - te
                if v["loss"] < best_val:
                    best_val, best_val_step = v["loss"], step
                evals.append((step, v["loss"]))
                for t in targets:
                    if time_to_target[str(t)] is None and v["loss"] <= t:
                        time_to_target[str(t)] = {"step": step, "tokens": (step + 1) * tokens_per_step,
                                                  "train_s": row["train_s"]}
            if log_f:
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
            if verbose:
                vl = f" val {row['val_loss']:.4f}" if "val_loss" in row else ""
                print(f"  step {step:5d}  loss {loss_v:.4f}{vl}  lr {opt.lr:.2e}  |g| {gnorm:.3f}")

        if out_dir and (step + 1) in ckpt_steps:
            te = time.time()
            _save_checkpoint(model, named, data, cfg, out_dir, step + 1, (step + 1) * tokens_per_step,
                             time.time() - t0 - eval_time)
            eval_time += time.time() - te

    wall = time.time() - t0
    steps_done = step + (0 if diverged else 1)
    final = None if diverged else evaluate(model, data, cfg)
    final_train = None if diverged or not cfg["train_eval"] else evaluate(model, data, cfg, split="train")
    nan = float("nan")
    param_bytes = sum(q.data.nbytes for q in params.values())
    summary = {
        # same keys as before (older runs have these too)
        "final_val_loss": final["loss"] if final else nan,
        "best_val_loss": best_val if math.isfinite(best_val) else nan,
        "final_train_ema": train_ema,
        "diverged": diverged,
        "steps": steps_done,
        "tokens": steps_done * tokens_per_step,
        "wall_time_s": wall,
        "tokens_per_s": steps_done * tokens_per_step / max(wall, 1e-9),
        "n_params": int(model.n_params()),
        "optimizer_state_bytes": opt.state_bytes(),
        "lca": lca,
        "evals": evals,
        # added metrics
        "metrics_version": 2,
        "best_val_step": best_val_step,
        "final_bpc": final["bpc"] if final else nan,
        "final_ppl": final["ppl"] if final else nan,
        "final_acc": final["acc"] if final else nan,
        "final_train_eval_loss": final_train["loss"] if final_train else nan,
        "generalization_gap": final["loss"] - final_train["loss"] if final and final_train else nan,
        "train_time_s": wall - eval_time,
        "train_tokens_per_s": steps_done * tokens_per_step / max(wall - eval_time, 1e-9),
        "clip_rate": clipped_total / max(steps_done, 1),
        "peak_mem_mb": peak_mem or nan,
        "model_mem_mb": (2 * param_bytes + opt.state_bytes()) / 2 ** 20,  # weights + grads + optimizer state
        "time_to_target": time_to_target,
    }
    if not diverged and cfg["final_eval"]:
        summary["final_eval"] = final_evaluation(model, data, cfg)

    if out_dir:
        log_f.close()
        if cfg["save_model"] and not diverged:
            arrays = {n: to_numpy(q.data) for n, q in named}
            np.savez(os.path.join(out_dir, "model.npz"), **arrays,
                     __chars__=np.array(json.dumps(data.chars)))
        json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=2)
        if cfg["write_report"]:
            try:
                from .report import run_report
                run_report(out_dir)
            except Exception as e:  # a report problem must never lose a finished run
                print(f"  (report not written: {e!r})")
    return summary


def _save_checkpoint(model, named, data, cfg, out_dir, step, tokens, train_s):
    """Weights + validation metrics + fixed-prompt samples at one point of training.
    Uses its own random streams, so training continues exactly as without it."""
    ck_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    np.savez(os.path.join(ck_dir, f"step_{step:06d}.npz"), **{n: to_numpy(q.data) for n, q in named},
             __chars__=np.array(json.dumps(data.chars)))
    v = evaluate(model, data, cfg)
    prompts = (cfg["gen_prompts"] or M.DEFAULT_PROMPTS)[:3]
    samples = []
    for i, p in enumerate(prompts):
        ids, _ = M.generate(model, cfg["ctx"], data.encode(p), cfg["checkpoint_sample_chars"],
                            cfg["gen_temperature"], cfg["gen_top_k"], seed=i)
        samples.append({"prompt": p, "text": data.decode(ids)})
    row = {"step": step, "tokens": tokens, "train_s": train_s, "val_loss": v["loss"], "val_bpc": v["bpc"],
           "val_acc": v["acc"], "samples": samples}
    with open(os.path.join(out_dir, "checkpoints.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def load_model(run_dir, weights=None):
    """Rebuild a trained model from a run directory (optionally from a checkpoint .npz)."""
    cfg = {**DEFAULTS, **json.load(open(os.path.join(run_dir, "config.json")))}
    blob = np.load(weights or os.path.join(run_dir, "model.npz"))
    chars = json.loads(str(blob["__chars__"]))
    model = GPT(cfg, len(chars), seed=0)
    for n, q in model.named_parameters():
        q.data = xp.asarray(blob[n])
    return model, cfg, chars
