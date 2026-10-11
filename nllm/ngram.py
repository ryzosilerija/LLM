"""Character n-gram language models as reference baselines.

Interpolated Witten-Bell smoothing:

    P_k(c | h) = (count(h, c) + N1(h) * P_{k-1}(c | h')) / (count(h) + N1(h))

where h is the previous k-1 characters, h' the previous k-2, and N1(h) the
number of distinct characters seen after h. Unseen contexts back off to the
lower order. P_1 is add-one smoothed. Each order is a proper probability
distribution, so its bits per character is directly comparable to the
transformer's on the same text.
"""
import json
import os

import numpy as np


def _keys(a, k, V):
    n = len(a) - k + 1
    key = np.zeros(n, dtype=np.int64)
    for j in range(k):
        key = key * V + a[j:j + n]
    return key


class WittenBell:
    def __init__(self, train, V, max_order=8):
        a = np.asarray(train).astype(np.int64)
        while max_order > 1 and float(V) ** max_order >= 2 ** 62:
            max_order -= 1
        self.V, self.max_order = V, max_order
        self.uni = np.bincount(a, minlength=V).astype(np.float64)
        self.N = len(a)
        self.tables = {}
        for k in range(2, max_order + 1):
            full, counts = np.unique(_keys(a, k, V), return_counts=True)
            ctx = full // V  # sorted, since full is sorted
            ctx_u, first = np.unique(ctx, return_index=True)
            total = np.add.reduceat(counts, first).astype(np.float64)
            distinct = np.diff(np.append(first, len(full))).astype(np.float64)
            self.tables[k] = (full, counts.astype(np.float64), ctx_u, total, distinct)

    def probs(self, seq):
        """{order: probability of each character from position max_order-1 on}."""
        a = np.asarray(seq).astype(np.int64)
        t = np.arange(self.max_order - 1, len(a))
        p = (self.uni[a[t]] + 1) / (self.N + self.V)
        out = {1: p}
        for k in range(2, self.max_order + 1):
            full_t, counts, ctx_u, total, distinct = self.tables[k]
            full = _keys(a, k, self.V)[t - k + 1]
            ctx = full // self.V
            i = np.minimum(np.searchsorted(ctx_u, ctx), len(ctx_u) - 1)
            seen = ctx_u[i] == ctx
            j = np.minimum(np.searchsorted(full_t, full), len(full_t) - 1)
            c = np.where(full_t[j] == full, counts[j], 0.0)
            T, D = np.where(seen, total[i], 0.0), np.where(seen, distinct[i], 0.0)
            p = np.where(seen, (c + D * p) / np.maximum(T + D, 1e-12), p)
            out[k] = p
        return out

    def bpc(self, seq):
        return {k: float(-np.log2(p).mean()) for k, p in self.probs(seq).items()}


def baselines(data, max_order=8):
    """BPC of n-gram models of every order on val (and test if present).
    Cached as ngram_baselines.json in the data directory."""
    path = os.path.join(data.data_dir, "ngram_baselines.json")
    if os.path.exists(path):
        cached = json.load(open(path))
        if cached.get("train_chars") == int(len(data.train)) and all(s in cached for s in _splits(data)):
            return cached
    model = WittenBell(data.train, data.vocab_size, max_order)
    out = {"train_chars": int(len(data.train)), "method": "interpolated Witten-Bell"}
    for s in _splits(data):
        out[s] = {str(k): v for k, v in model.bpc(data.split_array(s)).items()}
    json.dump(out, open(path, "w"), indent=2)
    return out


def _splits(data):
    return [s for s in ("val", "test") if data.has(s)]


def best(bl, split="val"):
    """(order, bpc) of the best n-gram order on a split."""
    if not bl or split not in bl:
        return None
    k, v = min(bl[split].items(), key=lambda kv: kv[1])
    return int(k), v
