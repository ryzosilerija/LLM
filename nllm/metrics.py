"""Evaluation metrics.

  eval_batches        loss / top-1 accuracy / BPC / perplexity on fixed random batches
  full_split_eval     the same over an entire split (sliding windows)
  generate            sampling, with timing (generation latency)
  text_stats          diversity and repetition of generated text
  ending_choice       story-ending choice benchmark (pick the real next sentence)
  inference_speed     batch throughput and single-sequence latency
  memory_mb           current process memory (or GPU pool memory with CuPy)
  window_overlap      n-gram window overlap, for leakage / memorization checks

Units: loss is mean negative log-likelihood per character in nats.
bits per character (BPC) = loss / ln 2.  perplexity = exp(loss).
"""
import math
import os
import re
import sys
import time

import numpy as np

from .backend import BACKEND, to_numpy, xp
from .tensor import no_grad

LN2 = math.log(2)

DEFAULT_PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "The dog was very happy because",
    "Tom wanted to play outside, but",
    "In a big forest, there",
]


def bpc(loss):
    return loss / LN2


def ppl(loss):
    return math.exp(loss) if loss < 50 else float("inf")


def _sync():
    if BACKEND == "cupy":
        xp.cuda.Device().synchronize()


def _token_nll_and_correct(model, x):
    """Per-position NLL and argmax-correctness for targets y, without autograd."""
    logits = model(x).data
    return logits


def _score(logits, y):
    m = logits.max(-1, keepdims=True)
    lse = m[..., 0] + xp.log(xp.exp(logits - m).sum(-1))
    tgt = xp.take_along_axis(logits, y[..., None], -1)[..., 0]
    nll = to_numpy(lse - tgt).astype(np.float64)
    correct = to_numpy(logits.argmax(-1) == y)
    return nll, correct


def eval_batches(model, data, split, batch_size, ctx, n_batches, seed):
    """Mean metrics over n_batches random windows drawn with a fixed seed,
    so every run is scored on exactly the same batches."""
    rng = np.random.default_rng(seed)
    nll_sum, correct_sum, count = 0.0, 0, 0
    with no_grad():
        for _ in range(n_batches):
            x, y = data.batch(split, batch_size, ctx, rng)
            nll, correct = _score(model(x).data, y)
            nll_sum += nll.sum()
            correct_sum += int(correct.sum())
            count += nll.size
    loss = nll_sum / count
    return {"loss": loss, "acc": correct_sum / count, "bpc": bpc(loss), "ppl": ppl(loss)}


def full_split_eval(model, data, split, ctx, stride=None, batch_size=64):
    """Evaluate every character of a split.

    Windows of length ctx start every `stride` characters. The first window
    scores all its positions; later windows score only their last `stride`
    positions, so each character is scored once and (for stride < ctx) with
    at least ctx - stride characters of context."""
    src = data.split_array(split)
    stride = stride or ctx
    starts = np.arange(0, len(src) - ctx - 1, stride)
    nll_sum, correct_sum, count = 0.0, 0, 0
    with no_grad():
        for b in range(0, len(starts), batch_size):
            ss = starts[b:b + batch_size]
            x = np.stack([src[s:s + ctx] for s in ss]).astype(np.int64)
            y = np.stack([src[s + 1:s + ctx + 1] for s in ss]).astype(np.int64)
            nll, correct = _score(model(xp.asarray(x)).data, xp.asarray(y))
            k0 = np.where(ss == 0, 0, ctx - stride)
            mask = np.arange(ctx)[None, :] >= k0[:, None]
            nll_sum += nll[mask].sum()
            correct_sum += int(correct[mask].sum())
            count += int(mask.sum())
    loss = nll_sum / max(count, 1)
    return {"loss": loss, "acc": correct_sum / max(count, 1), "bpc": bpc(loss), "ppl": ppl(loss),
            "chars": count, "stride": int(stride)}


def generate(model, ctx, prompt_ids, n_chars, temperature=0.8, top_k=20, seed=0):
    """Autoregressive sampling. Returns (generated ids, seconds)."""
    rng = np.random.default_rng(seed)
    ids = list(prompt_ids) or [0]
    start = len(ids)
    t0 = time.perf_counter()
    with no_grad():
        for _ in range(n_chars):
            window = np.asarray([ids[-ctx:]], dtype=np.int64)
            logits = to_numpy(model(xp.asarray(window)).data[0, -1]).astype(np.float64)
            if temperature <= 0:
                ids.append(int(logits.argmax()))
                continue
            logits /= temperature
            if top_k and top_k < len(logits):
                cut = np.partition(logits, -top_k)[-top_k]
                logits[logits < cut] = -np.inf
            p = np.exp(logits - logits.max())
            ids.append(int(rng.choice(len(p), p=p / p.sum())))
    _sync()
    return ids[start:], time.perf_counter() - t0


def _ngrams(words, n):
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def text_stats(texts, known_words=None):
    """Diversity / repetition of generated texts (word level).

      distinct_n    unique n-grams / all n-grams, pooled over samples (higher = more diverse)
      rep_4         share of word 4-grams that repeat an earlier 4-gram in the same
                    sample (higher = more looping)
      real_word     share of generated words that occur in the training text
                    (a char-level model can invent non-words)
    """
    words = [re.findall(r"\w+", t.lower()) for t in texts]
    out = {}
    for n in (1, 2, 3):
        grams = [g for w in words for g in _ngrams(w, n)]
        out[f"distinct_{n}"] = len(set(grams)) / len(grams) if grams else 0.0
    reps = []
    for w in words:
        g = _ngrams(w, 4)
        if g:
            seen, rep = set(), 0
            for x in g:
                rep += x in seen
                seen.add(x)
            reps.append(rep / len(g))
    out["rep_4"] = float(np.mean(reps)) if reps else 0.0
    if known_words is not None:
        allw = [x for w in words for x in w]
        out["real_word"] = sum(x in known_words for x in allw) / len(allw) if allw else 0.0
    out["words"] = sum(len(w) for w in words)
    return out


def ending_choice_items(data, split, ctx, n_items=200, n_options=4, end_len=(20, 48), seed=0):
    """Build 'pick the real next sentence' items from a split.

    Each item: ~ctx - 49 characters of story context, the sentence that really
    follows it, and n_options - 1 distractor sentences of similar length from
    elsewhere in the split. Chance accuracy is 1 / n_options."""
    text = data.split_text(split)
    ids = data.split_array(split)
    lo, hi = end_len
    ctx_len = ctx - hi - 1
    spans = [(m.start(), m.end()) for m in re.finditer(r"[^.!?\n]+[.!?]", text)
             if lo <= m.end() - m.start() <= hi]
    rng = np.random.default_rng(seed)
    candidates = [s for s in spans if s[0] >= ctx_len and "\n\n" not in text[s[0] - ctx_len:s[0]]]
    if len(candidates) < n_options + 1:
        return []
    picks = rng.choice(len(candidates), size=min(n_items, len(candidates)), replace=False)
    items = []
    for p in picks:
        s, e = candidates[p]
        options = [ids[s:e]]
        while len(options) < n_options:
            ds, de = spans[rng.integers(len(spans))]
            if (ds, de) != (s, e):
                options.append(ids[ds:de])
        order = rng.permutation(n_options)
        items.append({"context": ids[s - ctx_len:s], "options": [options[i] for i in order],
                      "answer": int(np.where(order == 0)[0][0])})
    return items


def ending_choice(model, items, ctx, neutral_id=None, batch_size=64):
    """Accuracy of picking the real next sentence.

    acc_raw:  option with the highest mean log-likelihood per character given
              the context. Biased toward sentences that are predictable anyway
              ("She was happy.").
    acc:      calibrated (PMI) score = log-likelihood given the context minus
              log-likelihood given a neutral one-character context, i.e. how
              much the context helps (Holtzman et al., 2021, "Surface Form
              Competition"). This isolates whether the model uses the story.
    """
    if not items:
        return {"acc": float("nan"), "acc_raw": float("nan"), "items": 0, "chance": float("nan")}
    neutral = np.asarray([neutral_id if neutral_id is not None else 0], dtype=np.int64)
    seqs, meta = [], []
    for k, it in enumerate(items):
        for j, opt in enumerate(it["options"]):
            seqs.append((np.concatenate([it["context"], opt]).astype(np.int64), len(it["context"]), len(opt), k, j, 0))
            seqs.append((np.concatenate([neutral, opt]).astype(np.int64), 1, len(opt), k, j, 1))
    shape = (len(items), len(items[0]["options"]))
    cond, uncond = np.full(shape, -np.inf), np.full(shape, -np.inf)
    with no_grad():
        for b in range(0, len(seqs), batch_size):
            chunk = seqs[b:b + batch_size]
            L = max(len(s[0]) for s in chunk) - 1
            x = np.zeros((len(chunk), L), dtype=np.int64)
            y = np.zeros((len(chunk), L), dtype=np.int64)
            for i, (s, *_r) in enumerate(chunk):
                x[i, :len(s) - 1] = s[:-1]
                y[i, :len(s) - 1] = s[1:]
            nll, _ = _score(model(xp.asarray(x)).data, xp.asarray(y))
            for i, (_s, c, n, k, j, is_uncond) in enumerate(chunk):
                ll = -nll[i, c - 1:c - 1 + n].mean()
                (uncond if is_uncond else cond)[k, j] = ll
    answers = np.array([it["answer"] for it in items])
    return {"acc": float(((cond - uncond).argmax(1) == answers).mean()),
            "acc_raw": float((cond.argmax(1) == answers).mean()),
            "items": len(items), "chance": 1.0 / shape[1]}


def inference_speed(model, ctx, vocab_size, batch_size=32, reps=5):
    """Forward-pass throughput (chars/s) at full context, batch and single."""
    rng = np.random.default_rng(0)
    out = {}
    with no_grad():
        for name, b in (("batch", batch_size), ("single", 1)):
            x = xp.asarray(rng.integers(0, vocab_size, size=(b, ctx)).astype(np.int64))
            model(x)
            _sync()
            t0 = time.perf_counter()
            for _ in range(reps):
                model(x)
            _sync()
            dt = (time.perf_counter() - t0) / reps
            out[f"{name}_ms"] = dt * 1000
            out[f"{name}_chars_per_s"] = b * ctx / dt
    return out


def memory_mb():
    """Current memory: GPU pool with CuPy, else process resident memory."""
    try:
        if BACKEND == "cupy":
            return xp.get_default_memory_pool().used_bytes() / 2 ** 20
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in (
                        "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage",
                        "PagefileUsage", "PeakPagefileUsage")]

            k32 = ctypes.WinDLL("kernel32")
            k32.GetCurrentProcess.restype = wintypes.HANDLE
            fn = k32.K32GetProcessMemoryInfo
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if fn(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 2 ** 20
            return None
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2 ** 20
    except Exception:
        return None


def window_hashes(arr, L):
    """64-bit polynomial hash of every length-L window (wraps on overflow)."""
    a = np.asarray(arr).astype(np.uint64)
    n = len(a) - L + 1
    if n <= 0:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    base = np.uint64(1000003)
    with np.errstate(over="ignore"):
        for j in range(L):
            h = h * base + a[j:j + n]
    return h


def window_overlap(query, reference_hashes, L):
    """Fraction of length-L windows of `query` that occur verbatim in the
    reference (whose sorted unique hashes are given)."""
    q = window_hashes(query, L)
    if len(q) == 0:
        return float("nan")
    return float(np.isin(q, reference_hashes, assume_unique=False).mean())


def longest_match(query, reference_hashes_by_len):
    """Longest L (from the given lengths) such that some window of `query`
    appears verbatim in the reference."""
    best = 0
    for L in sorted(reference_hashes_by_len):
        q = window_hashes(query, L)
        if len(q) and np.isin(q, reference_hashes_by_len[L]).any():
            best = L
    return best
