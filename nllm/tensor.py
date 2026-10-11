"""A small reverse-mode autograd engine.

Every operation creates a new Tensor that remembers its parents and a
closure computing the parents' gradients from its own. `backward()`
walks the graph in reverse topological order.

Rule used throughout: gradients are never modified in place, so two
tensors may safely share the same gradient array.
"""
import numpy as np

from .backend import xp, scatter_add

_DEFAULT_DTYPE = [np.float32]
_GRAD_ENABLED = [True]


def set_default_dtype(dtype):
    _DEFAULT_DTYPE[0] = dtype


def get_default_dtype():
    return _DEFAULT_DTYPE[0]


class no_grad:
    """Context manager that disables graph construction (used for eval)."""

    def __enter__(self):
        self.prev = _GRAD_ENABLED[0]
        _GRAD_ENABLED[0] = False

    def __exit__(self, *exc):
        _GRAD_ENABLED[0] = self.prev


def _unbroadcast(grad, shape):
    """Sum `grad` down to `shape`, undoing numpy broadcasting."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, s in enumerate(shape):
        if s == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


class Tensor:
    __array_priority__ = 1000

    def __init__(self, data, requires_grad=False, _parents=(), _op=""):
        if not isinstance(data, xp.ndarray):
            data = xp.asarray(data, dtype=get_default_dtype())
        self.data = data
        self.grad = None
        self.requires_grad = requires_grad
        self._parents = _parents
        self._backward = None
        self._op = _op

    # ------------------------------------------------------------ basics
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    def __repr__(self):
        return f"Tensor(shape={self.shape}, op={self._op!r}, requires_grad={self.requires_grad})"

    def item(self):
        return float(self.data)

    def _accum(self, g):
        if self.requires_grad:
            self.grad = g if self.grad is None else self.grad + g

    def backward(self, grad=None):
        topo, seen = [], set()
        stack = [(self, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                topo.append(node)
                continue
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.append((node, True))
            for p in node._parents:
                if id(p) not in seen:
                    stack.append((p, False))
        self.grad = xp.ones_like(self.data) if grad is None else grad
        for node in reversed(topo):
            if node._backward is not None and node.grad is not None:
                node._backward(node.grad)

    # --------------------------------------------------------- operators
    def __add__(self, o):
        return add(self, o)

    __radd__ = __add__

    def __mul__(self, o):
        return mul(self, o)

    __rmul__ = __mul__

    def __sub__(self, o):
        return add(self, neg(_lift(o, self)))

    def __rsub__(self, o):
        return add(_lift(o, self), neg(self))

    def __neg__(self):
        return neg(self)

    def __truediv__(self, o):
        return div(self, o)

    def __rtruediv__(self, o):
        return div(_lift(o, self), self)

    def __pow__(self, p):
        return power(self, p)

    def __matmul__(self, o):
        return matmul(self, o)

    def __getitem__(self, idx):
        return getitem(self, idx)

    def sum(self, axis=None, keepdims=False):
        return tsum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return tmean(self, axis, keepdims)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        return reshape(self, shape)

    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        return transpose(self, axes)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def tanh(self):
        return tanh(self)


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)


def _lift(x, like):
    if isinstance(x, Tensor):
        return x
    return Tensor(xp.asarray(x, dtype=like.data.dtype))


def _make(data, parents, op):
    rg = _GRAD_ENABLED[0] and any(p.requires_grad for p in parents)
    return Tensor(data, requires_grad=rg, _parents=parents if rg else (), _op=op)


# ===================================================================== ops
def add(a, b):
    a, b = _lift(a, b if isinstance(b, Tensor) else a), _lift(b, a if isinstance(a, Tensor) else b)
    out = _make(a.data + b.data, (a, b), "add")
    if out.requires_grad:
        def bw(g):
            a._accum(_unbroadcast(g, a.shape))
            b._accum(_unbroadcast(g, b.shape))
        out._backward = bw
    return out


def neg(a):
    out = _make(-a.data, (a,), "neg")
    if out.requires_grad:
        out._backward = lambda g: a._accum(-g)
    return out


def mul(a, b):
    a, b = _lift(a, b if isinstance(b, Tensor) else a), _lift(b, a if isinstance(a, Tensor) else b)
    out = _make(a.data * b.data, (a, b), "mul")
    if out.requires_grad:
        def bw(g):
            if a.requires_grad:
                a._accum(_unbroadcast(g * b.data, a.shape))
            if b.requires_grad:
                b._accum(_unbroadcast(g * a.data, b.shape))
        out._backward = bw
    return out


def div(a, b):
    a, b = _lift(a, b if isinstance(b, Tensor) else a), _lift(b, a if isinstance(a, Tensor) else b)
    out = _make(a.data / b.data, (a, b), "div")
    if out.requires_grad:
        def bw(g):
            if a.requires_grad:
                a._accum(_unbroadcast(g / b.data, a.shape))
            if b.requires_grad:
                b._accum(_unbroadcast(-g * a.data / (b.data * b.data), b.shape))
        out._backward = bw
    return out


def power(a, p):
    """Elementwise a**p for a Python scalar exponent p."""
    x = a.data
    if p == 2:  # fast paths: numpy's generic float pow is very slow
        y = x * x
        dy = lambda: 2.0 * x
    elif p == -0.5:
        y = 1.0 / xp.sqrt(x)
        dy = lambda: -0.5 * y / x
    elif p == 0.5:
        y = xp.sqrt(x)
        dy = lambda: 0.5 / y
    else:
        y = x ** p
        dy = lambda: p * x ** (p - 1)
    out = _make(y, (a,), "pow")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * dy())
    return out


def matmul(a, b):
    out = _make(a.data @ b.data, (a, b), "matmul")
    if out.requires_grad:
        def bw(g):
            if a.requires_grad:
                ga = g @ xp.swapaxes(b.data, -1, -2)
                a._accum(_unbroadcast(ga, a.shape))
            if b.requires_grad:
                if b.ndim == 2:
                    a2 = a.data.reshape(-1, a.shape[-1])
                    g2 = g.reshape(-1, g.shape[-1])
                    b._accum(a2.T @ g2)
                else:
                    gb = xp.swapaxes(a.data, -1, -2) @ g
                    b._accum(_unbroadcast(gb, b.shape))
        out._backward = bw
    return out


def exp(a):
    e = xp.exp(a.data)
    out = _make(e, (a,), "exp")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * e)
    return out


def log(a):
    out = _make(xp.log(a.data), (a,), "log")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g / a.data)
    return out


def tanh(a):
    t = xp.tanh(a.data)
    out = _make(t, (a,), "tanh")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * (1.0 - t * t))
    return out


def sigmoid(a):
    s = 1.0 / (1.0 + xp.exp(-a.data))
    out = _make(s, (a,), "sigmoid")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * s * (1.0 - s))
    return out


def relu(a):
    mask = a.data > 0
    out = _make(a.data * mask, (a,), "relu")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * mask)
    return out


_GELU_C = float(np.sqrt(2.0 / np.pi))


def gelu(a):
    """GELU, tanh approximation (as in GPT-2)."""
    x = a.data
    inner = _GELU_C * (x + 0.044715 * (x * x * x))
    t = xp.tanh(inner)
    out = _make(0.5 * x * (1.0 + t), (a,), "gelu")
    if out.requires_grad:
        def bw(g):
            dinner = _GELU_C * (1.0 + 3 * 0.044715 * x * x)
            d = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * dinner
            a._accum(g * d)
        out._backward = bw
    return out


def silu(a):
    """SiLU / Swish: x * sigmoid(x)."""
    x = a.data
    s = 1.0 / (1.0 + xp.exp(-x))
    out = _make(x * s, (a,), "silu")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * (s + x * s * (1.0 - s)))
    return out


def tsum(a, axis=None, keepdims=False):
    out = _make(a.data.sum(axis=axis, keepdims=keepdims), (a,), "sum")
    if out.requires_grad:
        def bw(g):
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                axes = sorted(ax % a.ndim for ax in axes)
                for ax in axes:
                    g = xp.expand_dims(g, ax)
            a._accum(xp.broadcast_to(g, a.shape))
        out._backward = bw
    return out


def tmean(a, axis=None, keepdims=False):
    if axis is None:
        n = a.size
    else:
        axes = axis if isinstance(axis, tuple) else (axis,)
        n = 1
        for ax in axes:
            n *= a.shape[ax]
    return tsum(a, axis, keepdims) * (1.0 / n)


def reshape(a, shape):
    out = _make(a.data.reshape(shape), (a,), "reshape")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g.reshape(a.shape))
    return out


def transpose(a, axes):
    out = _make(xp.transpose(a.data, axes), (a,), "transpose")
    if out.requires_grad:
        inv = tuple(np.argsort(axes))
        out._backward = lambda g: a._accum(xp.transpose(g, inv))
    return out


def _is_basic_index(idx):
    items = idx if isinstance(idx, tuple) else (idx,)
    return all(isinstance(i, (slice, int, type(None))) or i is Ellipsis for i in items)


def getitem(a, idx):
    out = _make(a.data[idx], (a,), "getitem")
    if out.requires_grad:
        def bw(g):
            full = xp.zeros_like(a.data)
            if _is_basic_index(idx):
                full[idx] = g
            else:
                scatter_add(full, idx, g)
            a._accum(full)
        out._backward = bw
    return out


def embedding(weight, idx):
    """Row lookup: weight[idx]. idx is an integer array (no grad)."""
    out = _make(weight.data[idx], (weight,), "embedding")
    if out.requires_grad:
        def bw(g):
            full = xp.zeros_like(weight.data)
            scatter_add(full, idx.reshape(-1), g.reshape(-1, weight.shape[1]))
            weight._accum(full)
        out._backward = bw
    return out


def softmax(a, axis=-1):
    x = a.data - a.data.max(axis=axis, keepdims=True)
    e = xp.exp(x)
    y = e / e.sum(axis=axis, keepdims=True)
    out = _make(y, (a,), "softmax")
    if out.requires_grad:
        def bw(g):
            a._accum(y * (g - (g * y).sum(axis=axis, keepdims=True)))
        out._backward = bw
    return out


def masked_fill(a, mask, value):
    """Where mask is True, replace with `value` (constant, no gradient)."""
    out = _make(xp.where(mask, xp.asarray(value, dtype=a.data.dtype), a.data), (a,), "masked_fill")
    if out.requires_grad:
        keep = ~mask
        out._backward = lambda g: a._accum(_unbroadcast(g * keep, a.shape))
    return out


def cross_entropy(logits, targets):
    """Mean cross-entropy. logits (N, V), targets int array (N,)."""
    x = logits.data
    m = x.max(axis=1, keepdims=True)
    lse = m + xp.log(xp.exp(x - m).sum(axis=1, keepdims=True))
    n = x.shape[0]
    rows = xp.arange(n)
    loss = (lse[:, 0] - x[rows, targets]).mean()
    out = _make(xp.asarray(loss, dtype=x.dtype), (logits,), "cross_entropy")
    if out.requires_grad:
        def bw(g):
            p = xp.exp(x - lse)
            p[rows, targets] -= 1.0
            logits._accum(p * (g / n))
        out._backward = bw
    return out


def _rotate_half(x):
    h = x.shape[-1] // 2
    return xp.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _rotate_half_T(x):
    # transpose of the linear map above: (z1, z2) -> (z2, -z1)
    h = x.shape[-1] // 2
    return xp.concatenate([x[..., h:], -x[..., :h]], axis=-1)


def rope(a, cos, sin):
    """Rotary position embedding. a: (..., T, D); cos/sin: (T, D) constants."""
    out = _make(a.data * cos + _rotate_half(a.data) * sin, (a,), "rope")
    if out.requires_grad:
        out._backward = lambda g: a._accum(g * cos + _rotate_half_T(g * sin))
    return out
