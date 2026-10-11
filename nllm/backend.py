"""Array backend.

All tensor math goes through `xp`, which is numpy by default. Set the
environment variable NLLM_BACKEND=cupy to run the same code on a GPU
(e.g. on Kaggle). Autograd, layers and optimizers are written by hand;
the backend only supplies raw array operations.
"""
import os

import numpy as np

BACKEND = os.environ.get("NLLM_BACKEND", "numpy").lower()

if BACKEND == "cupy":
    import cupy as xp  # noqa: F401
    import cupyx

    def scatter_add(target, idx, src):
        cupyx.scatter_add(target, idx, src)

    def to_numpy(a):
        return xp.asnumpy(a)
else:
    xp = np

    def scatter_add(target, idx, src):
        np.add.at(target, idx, src)

    def to_numpy(a):
        return np.asarray(a)


def asarray(a, dtype=None):
    return xp.asarray(a, dtype=dtype)
