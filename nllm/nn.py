"""Layers and the configurable GPT-style transformer.

Swappable components (chosen by name in the config):
  activation: relu, gelu, silu, sqrelu, geglu, swiglu
  norm:       layernorm, rmsnorm, dyt, scalenorm, none
  positional: learned, sinusoidal, rope

All blocks are pre-norm:  x = x + attn(norm(x));  x = x + ffn(norm(x)).
"""
import math

import numpy as np

from . import tensor as T
from .backend import xp
from .positional import rope_tables, sinusoidal_table
from .tensor import Parameter, Tensor, get_default_dtype


def _init(rng, shape, std):
    return Parameter(xp.asarray(rng.normal(0.0, std, size=shape), dtype=get_default_dtype()))


def _const(shape, value):
    return Parameter(xp.full(shape, value, dtype=get_default_dtype()))


class Module:
    record = False  # when True, layers store telemetry for this forward pass

    def named_parameters(self, prefix=""):
        out = []
        for name, v in vars(self).items():
            full = f"{prefix}{name}"
            if isinstance(v, Parameter):
                out.append((full, v))
            elif isinstance(v, Module):
                out.extend(v.named_parameters(full + "."))
            elif isinstance(v, (list, tuple)):
                for i, m in enumerate(v):
                    if isinstance(m, Module):
                        out.extend(m.named_parameters(f"{full}.{i}."))
        return out

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for v in vars(self).values():
            if isinstance(v, Module):
                yield from v.modules()
            elif isinstance(v, (list, tuple)):
                for m in v:
                    if isinstance(m, Module):
                        yield from m.modules()

    def set_record(self, flag):
        for m in self.modules():
            m.record = flag

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None


class Linear(Module):
    """y = x @ W, with W of shape (in, out). No bias (as in modern LLMs)."""

    def __init__(self, d_in, d_out, rng, std=0.02):
        self.weight = _init(rng, (d_in, d_out), std)

    def __call__(self, x):
        return x @ self.weight


class Embedding(Module):
    def __init__(self, n, d, rng, std=0.02):
        self.weight = _init(rng, (n, d), std)

    def __call__(self, idx):
        return T.embedding(self.weight, idx)


# ------------------------------------------------------------------ norms
class LayerNorm(Module):
    def __init__(self, d, eps=1e-5):
        self.gain = _const((d,), 1.0)
        self.bias = _const((d,), 0.0)
        self.eps = eps

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        xc = x - mu
        var = (xc * xc).mean(axis=-1, keepdims=True)
        return xc * (var + self.eps) ** -0.5 * self.gain + self.bias


class RMSNorm(Module):
    def __init__(self, d, eps=1e-6):
        self.gain = _const((d,), 1.0)
        self.eps = eps

    def __call__(self, x):
        ms = (x * x).mean(axis=-1, keepdims=True)
        return x * (ms + self.eps) ** -0.5 * self.gain


class DyT(Module):
    """Dynamic Tanh (Zhu et al., 2025): tanh(alpha * x) * gain + bias."""

    # alpha0 = 50 ~ 1 / (init activation scale 0.02). The paper default 0.5
    # assumes O(1) activations; with GPT-2 init it shrinks the signal ~50x.
    def __init__(self, d, alpha0=50.0):
        self.alpha = _const((1,), alpha0)
        self.gain = _const((d,), 1.0)
        self.bias = _const((d,), 0.0)

    def __call__(self, x):
        return T.tanh(x * self.alpha) * self.gain + self.bias


class ScaleNorm(Module):
    """g * x / ||x||, one learned scalar g (Nguyen & Salazar, 2019)."""

    def __init__(self, d, eps=1e-5):
        self.g = _const((1,), math.sqrt(d))
        self.eps = eps

    def __call__(self, x):
        sq = (x * x).sum(axis=-1, keepdims=True)
        return x * (sq + self.eps) ** -0.5 * self.g


class Identity(Module):
    def __call__(self, x):
        return x


NORMS = {
    "layernorm": LayerNorm,
    "rmsnorm": RMSNorm,
    "dyt": DyT,
    "scalenorm": ScaleNorm,
    "none": lambda d: Identity(),
}


# ------------------------------------------------------------ activations
PLAIN_ACTS = {
    "relu": T.relu,
    "gelu": T.gelu,
    "silu": T.silu,
    "sqrelu": lambda x: T.relu(x) ** 2,
}
GATED_ACTS = {"geglu": T.gelu, "swiglu": T.silu}
ACTIVATIONS = sorted(list(PLAIN_ACTS) + list(GATED_ACTS))


def ffn_hidden(d_model, activation):
    """Hidden width. Gated FFNs use 3 matrices, so their width is 2/3 of
    4*d to keep the parameter count matched with plain FFNs."""
    if activation in GATED_ACTS:
        return int(round(8 * d_model / 3))
    return 4 * d_model


class FFN(Module):
    def __init__(self, d, activation, rng, resid_std):
        self.activation = activation
        h = ffn_hidden(d, activation)
        self.gated = activation in GATED_ACTS
        if self.gated:
            self.w_gate = Linear(d, h, rng)
            self.w_up = Linear(d, h, rng)
        else:
            self.w_up = Linear(d, h, rng)
        self.w_down = Linear(h, d, rng, std=resid_std)
        self.stats = {}

    def __call__(self, x):
        if self.gated:
            gate = GATED_ACTS[self.activation](self.w_gate(x))
            h = gate * self.w_up(x)
            probe = gate
        else:
            h = PLAIN_ACTS[self.activation](self.w_up(x))
            probe = h
        if self.record:
            a = probe.data.reshape(-1, probe.shape[-1])
            active = a > 1e-6
            self.stats = {
                # unit never active on this batch
                "dead_frac": float((~active.any(axis=0)).mean()),
                # fraction of activations that are (near) zero or negative
                "sparsity": float((~active).mean()),
            }
        return self.w_down(h)


# -------------------------------------------------------------- attention
class Attention(Module):
    def __init__(self, d, n_head, positional, rng, resid_std):
        assert d % n_head == 0
        self.n_head, self.hd = n_head, d // n_head
        self.positional = positional
        self.wqkv = Linear(d, 3 * d, rng)
        self.wo = Linear(d, d, rng, std=resid_std)
        self.stats = {}

    def __call__(self, x):
        B, Tn, C = x.shape
        H, D = self.n_head, self.hd
        qkv = self.wqkv(x)
        q = qkv[:, :, 0:C].reshape(B, Tn, H, D).transpose(0, 2, 1, 3)
        k = qkv[:, :, C:2 * C].reshape(B, Tn, H, D).transpose(0, 2, 1, 3)
        v = qkv[:, :, 2 * C:].reshape(B, Tn, H, D).transpose(0, 2, 1, 3)
        if self.positional == "rope":
            cos, sin = rope_tables(Tn, D, x.data.dtype)
            q, k = T.rope(q, cos, sin), T.rope(k, cos, sin)
        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(D))
        mask = xp.triu(xp.ones((Tn, Tn), dtype=bool), k=1)
        att = T.softmax(T.masked_fill(scores, mask, -1e9), axis=-1)
        if self.record:
            p = att.data
            ent = -(p * xp.log(p + 1e-12)).sum(axis=-1)  # (B, H, T)
            self.stats = {"attn_entropy_per_head": [float(e) for e in ent.mean(axis=(0, 2))]}
        y = (att @ v).transpose(0, 2, 1, 3).reshape(B, Tn, C)
        return self.wo(y)


class Block(Module):
    def __init__(self, cfg, rng, resid_std):
        d = cfg["d_model"]
        self.norm1 = NORMS[cfg["norm"]](d)
        self.attn = Attention(d, cfg["n_head"], cfg["positional"], rng, resid_std)
        self.norm2 = NORMS[cfg["norm"]](d)
        self.ffn = FFN(d, cfg["activation"], rng, resid_std)

    def __call__(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GPT(Module):
    def __init__(self, cfg, vocab_size, seed):
        rng = np.random.default_rng(seed)
        d, L = cfg["d_model"], cfg["n_layer"]
        self.cfg, self.vocab_size = cfg, vocab_size
        self.positional = cfg["positional"]
        # GPT-2 scaled init for layers that write into the residual stream
        resid_std = 0.02 / math.sqrt(2 * L)
        self.tok_emb = Embedding(vocab_size, d, rng)
        if self.positional == "learned":
            self.pos_emb = Embedding(cfg["ctx"], d, rng)
        self.blocks = [Block(cfg, rng, resid_std) for _ in range(L)]
        self.norm_f = NORMS[cfg["norm"]](d)
        self.head = Linear(d, vocab_size, rng)
        self.stats = {}

    def __call__(self, idx):
        B, Tn = idx.shape
        x = self.tok_emb(idx)
        if self.positional == "learned":
            if Tn > self.cfg["ctx"]:
                raise ValueError("learned positions cannot go beyond the training context")
            x = x + T.embedding(self.pos_emb.weight, xp.arange(Tn))
        elif self.positional == "sinusoidal":
            # The raw table has RMS ~0.71 and would drown out token embeddings
            # (init std 0.02). Scale it to the init scale of a learned position
            # table, so sinusoidal vs learned differs only in fixed vs trained.
            x = x + Tensor(sinusoidal_table(Tn, self.cfg["d_model"], x.data.dtype) * (0.02 / math.sqrt(0.5)))
        resid = []
        for blk in self.blocks:
            x = blk(x)
            if self.record:
                resid.append(float(xp.sqrt((x.data ** 2).sum(-1)).mean()))
        if self.record:
            self.stats = {"resid_norm_per_layer": resid}
        return self.head(self.norm_f(x))

    def n_params(self):
        return sum(p.size for p in self.parameters())

    def telemetry(self):
        """Collect stats recorded during the last forward with record=True."""
        out = dict(self.stats)
        out["ffn"] = [b.ffn.stats for b in self.blocks]
        out["attn"] = [b.attn.stats for b in self.blocks]
        return out


def layer_group(name):
    """Map a parameter name to a coarse group for per-layer telemetry."""
    if name.startswith("blocks."):
        return "block" + name.split(".")[1]
    if name.startswith(("tok_emb", "pos_emb")):
        return "embed"
    if name.startswith("norm_f"):
        return "final_norm"
    return "head"
