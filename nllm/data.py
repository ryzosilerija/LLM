"""Character-level dataset.

Files in a data directory:
  train.bin, val.bin   uint16 character ids (first max_chars of the corpus, split 90/10)
  test.bin             optional held-out test split, taken from text *after*
                       max_chars, so it never overlaps train/val and adding it
                       doesn't change them
  meta.json            vocabulary
"""
import json
import os

import numpy as np

from .backend import xp

_CACHE = {}
SPLITS = ("train", "val", "test")


class CharData:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        meta = json.load(open(os.path.join(data_dir, "meta.json"), encoding="utf-8"))
        self.chars = meta["chars"]
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.arrays = {}
        for s in SPLITS:
            p = os.path.join(data_dir, f"{s}.bin")
            if os.path.exists(p):
                self.arrays[s] = np.fromfile(p, dtype=np.uint16)
        self.train, self.val = self.arrays["train"], self.arrays["val"]
        self._text = {}
        self._words = None

    def has(self, split):
        return split in self.arrays

    def split_array(self, split):
        if split not in self.arrays:
            raise FileNotFoundError(
                f"no {split}.bin in {self.data_dir}. Run scripts/prepare_data.py again: it now also writes test.bin")
        return self.arrays[split]

    def split_text(self, split):
        if split not in self._text:
            self._text[split] = self.decode(self.split_array(split))
        return self._text[split]

    def train_words(self):
        """Set of words seen in training text (for the real-word rate)."""
        if self._words is None:
            import re
            self._words = set(re.findall(r"\w+", self.split_text("train").lower()))
        return self._words

    def encode(self, s):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.chars[i] for i in ids)

    def batch(self, split, batch_size, ctx, rng):
        src = self.split_array(split)
        starts = rng.integers(0, len(src) - ctx - 1, size=batch_size)
        x = np.stack([src[s:s + ctx] for s in starts]).astype(np.int64)
        y = np.stack([src[s + 1:s + ctx + 1] for s in starts]).astype(np.int64)
        return xp.asarray(x), xp.asarray(y)


def load(data_dir):
    key = os.path.abspath(data_dir)
    if key not in _CACHE:
        _CACHE[key] = CharData(data_dir)
    return _CACHE[key]


def prepare(text, out_dir, val_frac=0.1, test_text=None):
    """Write train/val from `text` and, if given, test from `test_text`
    (characters outside the train/val vocabulary are dropped from test)."""
    os.makedirs(out_dir, exist_ok=True)
    chars = sorted(set(text))
    if len(chars) > 65535:
        raise ValueError("too many distinct characters")
    stoi = {c: i for i, c in enumerate(chars)}
    ids = np.array([stoi[c] for c in text], dtype=np.uint16)
    n = int(len(ids) * (1 - val_frac))
    ids[:n].tofile(os.path.join(out_dir, "train.bin"))
    ids[n:].tofile(os.path.join(out_dir, "val.bin"))
    meta = {"chars": chars, "train_tokens": n, "val_tokens": len(ids) - n}
    if test_text:
        tids = np.array([stoi[c] for c in test_text if c in stoi], dtype=np.uint16)
        tids.tofile(os.path.join(out_dir, "test.bin"))
        meta["test_tokens"] = len(tids)
        meta["test_dropped_chars"] = len(test_text) - len(tids)
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False)
    for name in ("ngram_baselines.json", "leakage.json"):  # caches derived from the old files
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            os.remove(p)
    return meta
