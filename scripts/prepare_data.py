"""Prepare a character-level dataset.

  # TinyStories validation split (~20 MB download):
  python scripts/prepare_data.py --tinystories

  # any UTF-8 text file (e.g. Lithuanian Wikipedia text):
  python scripts/prepare_data.py --file my_corpus.txt --out data/mycorpus

The first --max-chars characters become train/val (90/10). The next
--test-chars characters become the held-out test split. Rerunning on the same
corpus rewrites identical train/val files, so it's safe to rerun just to add
test.bin to an existing dataset.
"""
import argparse
import os
import sys
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm.data import prepare  # noqa: E402

TINYSTORIES_URL = ("https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
                   "TinyStoriesV2-GPT4-valid.txt")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--tinystories", action="store_true", help="download TinyStories (validation split)")
    src.add_argument("--file", help="path to a UTF-8 text file")
    ap.add_argument("--out", default="data/tinystories")
    ap.add_argument("--max-chars", type=int, default=5_000_000,
                    help="characters used for train/val (5M is plenty for ~120K-param models)")
    ap.add_argument("--test-chars", type=int, default=500_000,
                    help="characters after --max-chars used as the held-out test split")
    args = ap.parse_args()

    if args.tinystories:
        os.makedirs(args.out, exist_ok=True)
        raw = os.path.join(args.out, "raw.txt")
        if not os.path.exists(raw):
            print(f"downloading {TINYSTORIES_URL}")
            urllib.request.urlretrieve(TINYSTORIES_URL, raw)
        text = open(raw, encoding="utf-8").read().replace("<|endoftext|>", "\n")
    else:
        text = open(args.file, encoding="utf-8").read()

    main_text = text[: args.max_chars]
    test_text = text[args.max_chars: args.max_chars + args.test_chars]
    if len(test_text) < args.test_chars // 10:
        print(f"WARNING: only {len(test_text):,} characters left after --max-chars for the test split. "
              "Lower --max-chars to get a proper test set.")
    meta = prepare(main_text, args.out, test_text=test_text or None)
    print(f"wrote {args.out}: vocab {len(meta['chars'])}, train {meta['train_tokens']:,}, "
          f"val {meta['val_tokens']:,}, test {meta.get('test_tokens', 0):,} chars")


if __name__ == "__main__":
    main()
