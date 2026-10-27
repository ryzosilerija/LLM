"""Each optimizer must make solid progress on a small least-squares
problem with 2D weight matrices (named like hidden layers so Muon,
NorMuon and Shampoo use their matrix paths).

Run:  python tests/test_optimizers.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from nllm import tensor as T  # noqa: E402
from nllm.backend import BACKEND, to_numpy, xp  # noqa: E402
from nllm.optim import OPTIMIZERS, inverse_root, newton_schulz  # noqa: E402
from nllm.tensor import Parameter, Tensor  # noqa: E402

LRS = {"sgd": 0.05, "adamw": 1e-2, "lion": 1e-3, "muon": 1e-2, "normuon": 1e-2,
       "sophia": 3e-3, "shampoo": 1e-2}


def make_problem(seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(256, 16)).astype(np.float32)
    W_true = rng.normal(size=(16, 16)).astype(np.float32) / 4
    H = np.tanh(X @ W_true)
    Y = H @ (rng.normal(size=(16, 8)).astype(np.float32) / 4)
    return Tensor(X), Tensor(Y)


def run(name, steps=300):
    X, Y = make_problem()
    rng = np.random.default_rng(1)
    w1 = Parameter(rng.normal(0, 0.3, size=(16, 16)).astype(np.float32))
    w2 = Parameter(rng.normal(0, 0.3, size=(16, 8)).astype(np.float32))
    params = [("blocks.0.w1", w1), ("blocks.0.w2", w2)]
    opt = OPTIMIZERS[name](params, lr=LRS[name], weight_decay=0.0)

    def loss_fn():
        pred = T.tanh(X @ w1) @ w2
        d = pred - Y
        return (d * d).mean(), pred

    first = None
    for _ in range(steps):
        w1.grad = w2.grad = None
        if opt.needs_hessian and opt.t % opt.hessian_every == 0:
            # GNB analogue for squared loss: sample targets from N(pred, 1)
            _, pred = loss_fn()
            ys = Tensor(pred.data + xp.asarray(np.random.default_rng(opt.t).normal(size=pred.shape).astype(np.float32)))
            pred2 = T.tanh(X @ w1) @ w2
            d = pred2 - ys
            (d * d).mean().backward()
            opt.update_hessian({"blocks.0.w1": w1.grad, "blocks.0.w2": w2.grad}, 256)
            w1.grad = w2.grad = None
        loss, _ = loss_fn()
        loss.backward()
        opt.step()
        first = loss.item() if first is None else first
    final = loss_fn()[0].item()
    return first, final


def test_all():
    for name in OPTIMIZERS:
        first, final = run(name)
        ratio = final / first
        status = "ok " if ratio < 0.2 and np.isfinite(final) else "FAIL"
        print(f"  {status} {name:<8} loss {first:.4f} -> {final:.4f}  ({ratio:.1%} of start)")
        assert status == "ok ", name


def test_newton_schulz():
    G = np.random.default_rng(0).normal(size=(32, 64)).astype(np.float32)
    s = np.linalg.svd(to_numpy(newton_schulz(xp.asarray(G))), compute_uv=False)
    print(f"  ok  newton-schulz singular values in [{s.min():.2f}, {s.max():.2f}]")
    assert 0.5 < s.min() and s.max() < 1.3


def test_inverse_root():
    A = np.random.default_rng(0).normal(size=(10, 10))
    M = A @ A.T + np.eye(10)
    R = to_numpy(inverse_root(xp.asarray(M), 4, damping=0.0))
    err = np.abs(np.linalg.matrix_power(R, 4) @ M - np.eye(10)).max()
    print(f"  ok  inverse 4th root error {err:.1e}")
    assert err < 1e-6


if __name__ == "__main__":
    print(f"backend: {BACKEND}")
    test_newton_schulz()
    test_inverse_root()
    test_all()
    print("all optimizer tests passed")
