"""Positional encoding tables (constants, no parameters)."""
import numpy as np

from .backend import xp

_cache = {}


def sinusoidal_table(T, d, dtype):
    """Original Transformer encoding: sin on even dims, cos on odd dims."""
    key = ("sin", T, d, np.dtype(dtype).str)
    if key not in _cache:
        pos = np.arange(T)[:, None]
        i = np.arange(0, d, 2)[None, :]
        angle = pos / (10000.0 ** (i / d))
        tab = np.zeros((T, d))
        tab[:, 0::2] = np.sin(angle)
        tab[:, 1::2] = np.cos(angle[:, : d // 2])
        _cache[key] = xp.asarray(tab, dtype=dtype)
    return _cache[key]


def rope_tables(T, head_dim, dtype, base=10000.0):
    """cos/sin tables of shape (T, head_dim) using the rotate-half layout."""
    key = ("rope", T, head_dim, np.dtype(dtype).str, base)
    if key not in _cache:
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2) / head_dim))
        freqs = np.outer(np.arange(T), inv_freq)
        emb = np.concatenate([freqs, freqs], axis=1)
        _cache[key] = (xp.asarray(np.cos(emb), dtype=dtype), xp.asarray(np.sin(emb), dtype=dtype))
    return _cache[key]
