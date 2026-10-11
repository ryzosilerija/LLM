"""Checks for the evaluation metrics, using small synthetic inputs.

Run:  python tests/test_metrics.py
"""
import math
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import data as D  # noqa: E402
from nllm import metrics as M  # noqa: E402
from nllm.ngram import WittenBell  # noqa: E402


class Tn:
    def __init__(self, a):
        self.data = a


class Uniform:
    """Predicts every character with equal probability."""

    def __init__(self, V):
        self.V = V

    def __call__(self, x):
        x = np.asarray(x)
        return Tn(np.zeros(x.shape + (self.V,)))


def make_data():
    rng = np.random.default_rng(0)
    words = ["the", "cat", "sat", "on", "a", "mat", "dog", "ran", "home", "happy"]
    text = ". ".join(" ".join(rng.choice(words, size=rng.integers(4, 9))).capitalize() for _ in range(4000)) + "."
    d = tempfile.mkdtemp()
    D.prepare(text[:60000], d, test_text=text[60000:80000])
    return D.CharData(d)


def test_units():
    assert abs(M.bpc(math.log(2)) - 1.0) < 1e-12
    assert abs(M.ppl(math.log(7)) - 7.0) < 1e-9
    print("  ok  BPC / perplexity conversions")


def test_uniform_model(data):
    V = data.vocab_size
    r = M.full_split_eval(Uniform(V), data, "val", ctx=32, stride=16)
    assert abs(r["bpc"] - math.log2(V)) < 1e-9, r
    expected = len(data.val) - 1 - (len(data.val) - 1 - 32) % 16  # every char scored once except the tail
    assert abs(r["chars"] - expected) <= 32, (r["chars"], expected)
    b = M.eval_batches(Uniform(V), data, "val", 4, 32, 3, seed=0)
    assert abs(b["loss"] - math.log(V)) < 1e-9
    print(f"  ok  uniform model scores log2(V) = {math.log2(V):.3f} BPC; {r['chars']:,} of {len(data.val):,} chars scored")


def test_stride_covers_each_char_once(data):
    V = data.vocab_size
    a = M.full_split_eval(Uniform(V), data, "val", ctx=32, stride=32)
    b = M.full_split_eval(Uniform(V), data, "val", ctx=32, stride=8)
    assert abs(a["chars"] - b["chars"]) <= 32, (a["chars"], b["chars"])
    print("  ok  sliding windows score each character once (stride 32 vs 8)")


def test_text_stats():
    t = M.text_stats(["the cat sat on the mat", "a dog a dog a dog a dog a dog"], {"the", "cat", "sat", "on", "mat"})
    assert abs(t["rep_4"] - (0 + 5 / 7) / 2) < 1e-9, t  # sample 2 loops: 5 of its 7 word 4-grams repeat
    assert 0 < t["distinct_2"] < 1
    assert abs(t["real_word"] - 6 / 16) < 1e-9, t
    print(f"  ok  text stats: rep-4 {t['rep_4']:.2f}, distinct-2 {t['distinct_2']:.2f}, real words {t['real_word']:.2f}")


def test_overlap():
    ref = np.arange(1000) % 50
    h = np.unique(M.window_hashes(ref, 16))
    assert M.window_overlap(ref[100:300], h, 16) == 1.0
    other = (np.arange(200) * 7 + 3) % 97
    assert M.window_overlap(other, h, 16) == 0.0
    assert M.longest_match(np.concatenate([ref[:40], other]), {L: np.unique(M.window_hashes(ref, L))
                                                               for L in (16, 32, 64)}) == 32
    print("  ok  window overlap and longest verbatim match")


def test_ngram_normalized(data):
    m = WittenBell(data.train, data.vocab_size, 4)
    ctx = data.val[:3].astype(np.int64)
    tot = sum(m.probs(np.append(ctx, c))[4][-1] for c in range(data.vocab_size))
    assert abs(tot - 1) < 1e-9, tot
    bpc = m.bpc(data.val)
    assert bpc[4] < bpc[1], bpc
    print(f"  ok  n-gram probabilities sum to 1; BPC 1-gram {bpc[1]:.2f} -> 4-gram {bpc[4]:.2f}")


def test_ending_choice(data):
    items = M.ending_choice_items(data, "val", 64, 50, end_len=(10, 30))
    assert items and all(len(it["options"]) == 4 for it in items)
    r = M.ending_choice(Uniform(data.vocab_size), items, 64)
    # a uniform model gives every option the same score -> argmax always picks option 0
    assert abs(r["acc"] - np.mean([it["answer"] == 0 for it in items])) < 1e-9
    print(f"  ok  story-ending items built ({len(items)}), scoring consistent")


if __name__ == "__main__":
    data = make_data()
    test_units()
    test_uniform_model(data)
    test_stride_covers_each_char_once(data)
    test_text_stats()
    test_overlap()
    test_ngram_normalized(data)
    test_ending_choice(data)
    print("all metric tests passed")
