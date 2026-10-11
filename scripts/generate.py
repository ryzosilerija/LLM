"""Generate text from a trained run.

  python scripts/generate.py results/runs/01_adamw_gelu_layernorm_learned_s1 --prompt "Once upon a time"
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm.metrics import generate  # noqa: E402
from nllm.train import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--prompt", default="Once upon a time")
    ap.add_argument("--n", type=int, default=400, help="characters to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="0 = greedy")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    model, cfg, chars = load_model(args.run_dir)
    stoi = {c: i for i, c in enumerate(chars)}
    ids, dt = generate(model, cfg["ctx"], [stoi[c] for c in args.prompt if c in stoi], args.n,
                       args.temperature, args.top_k, args.seed)
    print(args.prompt + "".join(chars[i] for i in ids))
    print(f"\n[{args.n} chars in {dt:.2f} s, {1000 * dt / max(args.n, 1):.1f} ms/char]")


if __name__ == "__main__":
    main()
