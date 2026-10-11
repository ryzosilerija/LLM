"""Numerical gradient checks: autograd vs central finite differences.

Run:  python tests/test_gradcheck.py
"""
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import tensor as T  # noqa: E402
from nllm.nn import GPT  # noqa: E402
from nllm.tensor import Parameter, Tensor, set_default_dtype  # noqa: E402

set_default_dtype(np.float64)
RNG = np.random.default_rng(0)
EPS, TOL = 1e-6, 1e-5


def numeric_grad(f, p):
    g = np.zeros_like(p.data)
    it = np.nditer(p.data, flags=["multi_index"])
    for _ in it:
        i = it.multi_index
        old = p.data[i]
        p.data[i] = old + EPS
        fp = f().item()
        p.data[i] = old - EPS
        fm = f().item()
        p.data[i] = old
        g[i] = (fp - fm) / (2 * EPS)
    return g


def check(name, f, params):
    for p in params:
        p.grad = None
    f().backward()
    worst = 0.0
    for p in params:
        num = numeric_grad(f, p)
        err = np.abs(num - p.grad).max() / max(1.0, np.abs(num).max())
        worst = max(worst, err)
    assert worst < TOL, f"{name}: relative error {worst:.2e}"
    print(f"  ok  {name:<32} err={worst:.1e}")


def P(*shape, positive=False):
    a = RNG.normal(size=shape)
    return Parameter(np.abs(a) + 0.5 if positive else a)


def test_ops():
    a, b, c = P(3, 4), P(3, 4), P(4)
    w, x3 = P(4, 5), P(2, 3, 4)
    pos = P(3, 4, positive=True)
    r = Tensor(RNG.normal(size=(3, 4)))  # fixed weights to reduce outputs to scalars
    rope_x = P(2, 3, 5, 4)
    cos, sin = np.cos(RNG.normal(size=(5, 4))), np.sin(RNG.normal(size=(5, 4)))
    mask = RNG.random((3, 4)) > 0.5
    idx = np.array([[0, 2, 2], [1, 0, 3]])
    emb = P(4, 3)
    logits, tgt = P(6, 5), np.array([0, 4, 2, 2, 1, 3])
    cases = {
        "add+broadcast": (lambda: ((a + c) * r).sum(), [a, c]),
        "sub/neg": (lambda: ((a - b) * r).sum(), [a, b]),
        "mul": (lambda: (a * b * r).sum(), [a, b]),
        "div": (lambda: (a / pos * r).sum(), [a, pos]),
        "pow": (lambda: (pos ** -0.5 * r).sum(), [pos]),
        "matmul 2d": (lambda: (a @ w).sum(), [a, w]),
        "matmul batched x 2d": (lambda: ((x3 @ w) ** 2).sum(), [x3, w]),
        "matmul batched x batched": (lambda: ((x3 @ x3.transpose(0, 2, 1)) ** 2).sum(), [x3]),
        "exp/log": (lambda: (T.log(pos) + T.exp(a) * r).sum(), [a, pos]),
        "tanh": (lambda: (T.tanh(a) * r).sum(), [a]),
        "sigmoid": (lambda: (T.sigmoid(a) * r).sum(), [a]),
        "relu": (lambda: (T.relu(a) * r).sum(), [a]),
        "gelu": (lambda: (T.gelu(a) * r).sum(), [a]),
        "silu": (lambda: (T.silu(a) * r).sum(), [a]),
        "sum/mean axis": (lambda: (a.sum(axis=0) * a.mean(axis=1, keepdims=True)).sum(), [a]),
        "reshape/transpose": (lambda: ((x3.reshape(3, 2, 4).transpose(2, 0, 1)) ** 2).sum(), [x3]),
        "getitem slice": (lambda: (a[:, 1:3] ** 2).sum(), [a]),
        "embedding": (lambda: (T.embedding(emb, idx) ** 2).sum(), [emb]),
        "softmax": (lambda: (T.softmax(a, axis=-1) * r).sum(), [a]),
        "masked_fill": (lambda: (T.masked_fill(a, mask, -5.0) * r).sum(), [a]),
        "cross_entropy": (lambda: T.cross_entropy(logits, tgt), [logits]),
        "rope": (lambda: (T.rope(rope_x, cos, sin) ** 2).sum(), [rope_x]),
    }
    for name, (f, ps) in cases.items():
        check(name, f, ps)


def test_models():
    """Full model gradient check for every activation, norm and positional encoding."""
    base = dict(d_model=8, n_layer=2, n_head=2, ctx=6)
    combos = (
        [("gelu", "layernorm", p) for p in ("learned", "sinusoidal", "rope")]
        + [(a, "rmsnorm", "rope") for a in ("relu", "silu", "sqrelu", "geglu", "swiglu")]
        + [("gelu", n, "learned") for n in ("rmsnorm", "dyt", "scalenorm", "none")]
    )
    idx = RNG.integers(0, 7, size=(2, 6))
    tgt = RNG.integers(0, 7, size=(12,))
    for act, norm, pos in combos:
        cfg = dict(base, activation=act, norm=norm, positional=pos)
        model = GPT(cfg, vocab_size=7, seed=1)
        # bigger init so the check exercises non-trivial values
        for _, p in model.named_parameters():
            p.data = p.data * 10 if p.ndim == 2 else p.data + RNG.normal(0, 0.1, size=p.shape)

        def f():
            return T.cross_entropy(model(idx).reshape(12, 7), tgt)

        model.zero_grad()
        f().backward()
        worst = 0.0
        for name, p in model.named_parameters():
            # check a random subset of entries per tensor to keep it fast
            g_auto = p.grad
            flat = p.data.reshape(-1)
            for j in RNG.choice(flat.size, size=min(6, flat.size), replace=False):
                old = flat[j]
                flat[j] = old + EPS
                fp = f().item()
                flat[j] = old - EPS
                fm = f().item()
                flat[j] = old
                num = (fp - fm) / (2 * EPS)
                worst = max(worst, abs(num - g_auto.reshape(-1)[j]) / max(1.0, abs(num)))
        assert worst < TOL, f"model {act}/{norm}/{pos}: err {worst:.2e}"
        print(f"  ok  model {act:>6}/{norm:<9}/{pos:<10} err={worst:.1e}")


if __name__ == "__main__":
    print("op gradient checks")
    test_ops()
    print("model gradient checks")
    test_models()
    print("all gradient checks passed")
