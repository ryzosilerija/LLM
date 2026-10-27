"""A CPU stand-in for CuPy that enforces CuPy's main rule: device arrays and
plain numpy arrays must never be mixed. Select it with NLLM_BACKEND=strict to
catch GPU-only bugs on a machine without a GPU. Not used for real training.

Every array created through this module is a StrictArray (a numpy subclass).
Any ufunc, numpy function or index that receives a plain numpy array raises
TypeError, just like CuPy would.
"""
import numpy as _np


class StrictArray(_np.ndarray):
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        _check(inputs, f"ufunc {ufunc.__name__}")
        _check(kwargs.get("out", ()), f"ufunc {ufunc.__name__} out")
        args = _unwrap(inputs)
        if "out" in kwargs:
            kwargs["out"] = _unwrap(kwargs["out"])
        res = getattr(ufunc, method)(*args, **kwargs)
        return None if method == "at" else _wrap(res)

    def __array_function__(self, func, types, args, kwargs):
        _check(args, func.__name__)
        _check(tuple(kwargs.values()), func.__name__)
        return _wrap(func(*_unwrap(args), **_unwrap(kwargs)))

    def __getitem__(self, idx):
        _check(idx if isinstance(idx, tuple) else (idx,), "indexing")
        return _wrap(_np.ndarray.__getitem__(self.view(_np.ndarray), _unwrap(idx)))

    def __setitem__(self, idx, value):
        _check(idx if isinstance(idx, tuple) else (idx,), "index assignment")
        _check((value,), "index assignment value")
        _np.ndarray.__setitem__(self.view(_np.ndarray), _unwrap(idx), _unwrap(value))


def _check(objs, where):
    for o in objs:
        if type(o) is _np.ndarray:
            raise TypeError(f"plain numpy array passed to {where}: CuPy would reject this "
                            "(convert with xp.asarray first)")
        if isinstance(o, (list, tuple)):
            _check(o, where)
        elif isinstance(o, dict):
            _check(tuple(o.values()), where)


def _unwrap(o):
    if isinstance(o, StrictArray):
        return o.view(_np.ndarray)
    if isinstance(o, tuple):
        return tuple(_unwrap(x) for x in o)
    if isinstance(o, list):
        return [_unwrap(x) for x in o]
    if isinstance(o, dict):
        return {k: _unwrap(v) for k, v in o.items()}
    return o


def _wrap(o):
    if isinstance(o, _np.ndarray) and not isinstance(o, StrictArray):
        return o.view(StrictArray)
    if isinstance(o, tuple):
        return tuple(_wrap(x) for x in o)
    if isinstance(o, list) and o and isinstance(o[0], _np.ndarray):
        return [_wrap(x) for x in o]
    return o


class _Proxy:
    def __init__(self, mod):
        self._mod = mod

    def __getattr__(self, name):
        if name == "ndarray":
            return StrictArray
        attr = getattr(self._mod, name)
        if isinstance(attr, type(_np)):
            return _Proxy(attr)
        if isinstance(attr, type) or not callable(attr):
            return attr

        def call(*args, **kwargs):
            if name in ("asarray", "array", "ascontiguousarray"):  # conversion is allowed
                return _wrap(attr(*_unwrap(args), **_unwrap(kwargs)))
            _check(args, name)
            _check(tuple(kwargs.values()), name)
            return _wrap(attr(*_unwrap(args), **_unwrap(kwargs)))
        return call


linalg = _Proxy(_np.linalg)
_main = _Proxy(_np)


def __getattr__(name):
    return getattr(_main, name)


def asnumpy(a):
    return _np.asarray(a.view(_np.ndarray) if isinstance(a, StrictArray) else a).copy()
