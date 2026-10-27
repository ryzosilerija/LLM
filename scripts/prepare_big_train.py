"""Build a dataset with a much larger training split but the SAME val and test
splits (and vocabulary) as an existing dataset, so models trained on it are
directly comparable with every earlier experiment.

  python scripts/prepare_big_train.py --base data/tinystories --out data/tinystories_big --train-chars 300000000
"""
import argparse
import json
import os
import shutil
import time
import urllib.request

import numpy as np

TRAIN_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
             "TinyStoriesV2-GPT4-train.txt")


def encode(text, chars):
    """Map text to vocabulary ids (vectorized). Returns (ids uint16, dropped count)."""
    cps = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    vocab = np.array([ord(c) for c in chars], dtype=np.uint32)
    order = np.argsort(vocab)
    sorted_vocab = vocab[order]
    pos = np.searchsorted(sorted_vocab, cps)
    pos = np.minimum(pos, len(sorted_vocab) - 1)
    known = sorted_vocab[pos] == cps
    ids = order[pos[known]].astype(np.uint16)
    return ids, int((~known).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="data/tinystories", help="existing dataset whose val/test/vocab are kept")
    ap.add_argument("--out", default="data/tinystories_big")
    ap.add_argument("--file", default=None, help="UTF-8 source text (default: download TinyStories train)")
    ap.add_argument("--train-chars", type=int, default=300_000_000)
    ap.add_argument("--chunk-chars", type=int, default=50_000_000, help="read the source in pieces of this size")
    args = ap.parse_args()

    meta = json.load(open(os.path.join(args.base, "meta.json"), encoding="utf-8"))
    for f in ("val.bin", "test.bin"):
        if not os.path.exists(os.path.join(args.base, f)):
            raise SystemExit(f"{args.base} has no {f}; run scripts/prepare_data.py first")
    os.makedirs(args.out, exist_ok=True)

    src = args.file
    if src is None:
        src = os.path.join(args.out, "raw_train.txt")
        if not os.path.exists(src):
            print(f"downloading {TRAIN_URL} (~2 GB) ...")
            t0 = time.time()
            urllib.request.urlretrieve(TRAIN_URL, src + ".part")
            os.replace(src + ".part", src)
            print(f"  done in {time.time() - t0:.0f} s")

    t0 = time.time()
    parts, total_in, dropped = [], 0, 0
    with open(src, encoding="utf-8") as f:
        while total_in < args.train_chars:
            text = f.read(min(args.chunk_chars, args.train_chars - total_in))
            if not text:
                break
            total_in += len(text)
            ids, d = encode(text.replace("<|endoftext|>", "\n"), meta["chars"])
            parts.append(ids)
            dropped += d
            print(f"  encoded {total_in:,} characters", flush=True)
    train = np.concatenate(parts)
    train.tofile(os.path.join(args.out, "train.bin"))
    for f in ("val.bin", "test.bin"):
        shutil.copyfile(os.path.join(args.base, f), os.path.join(args.out, f))
    new_meta = dict(meta, train_tokens=int(len(train)), train_source=os.path.basename(src),
                    train_dropped_chars=dropped, val_test_from=os.path.normpath(args.base))
    json.dump(new_meta, open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False)
    for name in ("ngram_baselines.json", "leakage.json"):
        p = os.path.join(args.out, name)
        if os.path.exists(p):
            os.remove(p)
    if total_in < args.train_chars:
        print(f"WARNING: the source only had {total_in:,} characters")
    print(f"wrote {args.out}: train {len(train):,} chars ({dropped:,} unknown characters dropped), "
          f"val/test copied from {args.base} [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
