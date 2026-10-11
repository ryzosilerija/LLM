"""Seven optimizers, written from scratch.

    sgd       SGD with momentum
    adamw     Adam with decoupled weight decay (Loshchilov & Hutter, 2019)
    lion      sign of interpolated momentum (Chen et al., 2023)
    muon      orthogonalized momentum via Newton-Schulz (Jordan et al., 2024)
    normuon   Muon + per-neuron second-moment normalization (Li et al., 2025)
    sophia    clipped diagonal-Hessian preconditioning, GNB estimator (Liu et al., 2023)
    shampoo   Kronecker-factored preconditioning with Adam grafting (Gupta et al., 2018)

Conventions
  * Weight decay is decoupled (p -= lr * wd * p) and applies only to
    parameters with ndim >= 2 (matrices and embeddings, not gains/biases).
  * Muon, NorMuon and Shampoo only make sense for 2D hidden weight
    matrices. Embeddings, the output head and 1D params use an internal
    AdamW with the same learning rate, as is standard practice.
  * Muon/NorMuon updates are rescaled to roughly match AdamW's update RMS
    (factor 0.2 * sqrt(max(m, n)), Liu et al., 2025), and Shampoo grafts
    its step size from Adam, so all three share AdamW's LR scale.
"""
import math

from .backend import xp


def decays(name, p):
    return p.ndim >= 2


def is_hidden_matrix(name, p):
    return p.ndim == 2 and name.startswith("blocks.")


class Optimizer:
    needs_hessian = False

    def __init__(self, named_params, lr, weight_decay=0.0):
        self.params = list(named_params)
        self.lr = lr
        self.wd = weight_decay
        self.state = {}
        self.t = 0

    def _st(self, name):
        return self.state.setdefault(name, {})

    def _decay(self, name, p):
        if self.wd and decays(name, p):
            p.data -= self.lr * self.wd * p.data

    def step(self):
        self.t += 1
        for name, p in self.params:
            if p.grad is not None:
                self._update(name, p, p.grad)

    def _update(self, name, p, g):
        raise NotImplementedError

    def state_bytes(self):
        total = 0
        for st in self.state.values():
            for v in st.values():
                if hasattr(v, "nbytes"):
                    total += v.nbytes
        return int(total)


def _adamw(opt, name, p, g, betas=(0.9, 0.95), eps=1e-8):
    st = opt._st(name)
    if "m" not in st:
        st["m"] = xp.zeros_like(p.data)
        st["v"] = xp.zeros_like(p.data)
        st["t"] = 0
    st["t"] += 1
    b1, b2 = betas
    st["m"] = b1 * st["m"] + (1 - b1) * g
    st["v"] = b2 * st["v"] + (1 - b2) * g * g
    mh = st["m"] / (1 - b1 ** st["t"])
    vh = st["v"] / (1 - b2 ** st["t"])
    opt._decay(name, p)
    p.data -= opt.lr * mh / (xp.sqrt(vh) + eps)


class SGD(Optimizer):
    def __init__(self, named_params, lr, weight_decay=0.0, momentum=0.9):
        super().__init__(named_params, lr, weight_decay)
        self.momentum = momentum

    def _update(self, name, p, g):
        st = self._st(name)
        st["buf"] = self.momentum * st["buf"] + g if "buf" in st else g.copy()
        self._decay(name, p)
        p.data -= self.lr * st["buf"]


class AdamW(Optimizer):
    def __init__(self, named_params, lr, weight_decay=0.0, betas=(0.9, 0.95)):
        super().__init__(named_params, lr, weight_decay)
        self.betas = betas

    def _update(self, name, p, g):
        _adamw(self, name, p, g, self.betas)


class Lion(Optimizer):
    def __init__(self, named_params, lr, weight_decay=0.0, betas=(0.9, 0.99)):
        super().__init__(named_params, lr, weight_decay)
        self.betas = betas

    def _update(self, name, p, g):
        st = self._st(name)
        m = st.get("m", xp.zeros_like(p.data))
        b1, b2 = self.betas
        direction = xp.sign(b1 * m + (1 - b1) * g)
        st["m"] = b2 * m + (1 - b2) * g
        self._decay(name, p)
        p.data -= self.lr * direction


# Quintic Newton-Schulz coefficients from Keller Jordan's Muon.
_NS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G, steps=5):
    """Approximately orthogonalize G (replace its singular values with ~1)."""
    a, b, c = _NS
    X = G / (xp.sqrt((G * G).sum()) + 1e-7)
    transposed = X.shape[0] > X.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transposed else X


class Muon(Optimizer):
    def __init__(self, named_params, lr, weight_decay=0.0, momentum=0.95, ns_steps=5):
        super().__init__(named_params, lr, weight_decay)
        self.momentum, self.ns_steps = momentum, ns_steps

    def _direction(self, name, O):
        return O

    def _update(self, name, p, g):
        if not is_hidden_matrix(name, p):
            return _adamw(self, name, p, g)
        st = self._st(name)
        st["buf"] = self.momentum * st["buf"] + g if "buf" in st else g.copy()
        u = g + self.momentum * st["buf"]  # Nesterov
        O = newton_schulz(u, self.ns_steps)
        O = self._direction(name, O)
        m, n = p.shape
        self._decay(name, p)
        p.data -= self.lr * 0.2 * math.sqrt(max(m, n)) * O


class NorMuon(Muon):
    """Muon followed by per-neuron (per output unit) second-moment
    normalization, so no single neuron dominates the update. Weights are
    stored as (in, out), so an output neuron is a column."""

    def __init__(self, named_params, lr, weight_decay=0.0, momentum=0.95, beta2=0.95, ns_steps=5):
        super().__init__(named_params, lr, weight_decay, momentum, ns_steps)
        self.beta2 = beta2

    def _direction(self, name, O):
        st = self._st(name)
        col_ms = (O * O).mean(axis=0)
        st["v"] = self.beta2 * st["v"] + (1 - self.beta2) * col_ms if "v" in st else col_ms
        On = O / (xp.sqrt(st["v"]) + 1e-8)
        # restore the orthogonalized update's overall RMS (~1/sqrt(max(m, n)))
        target = xp.sqrt((O * O).sum())
        return On * (target / (xp.sqrt((On * On).sum()) + 1e-12))


class Sophia(Optimizer):
    """Sophia-G. The trainer calls update_hessian() every `hessian_every`
    steps with gradients computed on labels sampled from the model's own
    predictions (Gauss-Newton-Bartlett estimator: h ~ B * g_hat^2)."""

    needs_hessian = True

    def __init__(self, named_params, lr, weight_decay=0.0, betas=(0.965, 0.99), rho=0.04,
                 hessian_every=10):
        super().__init__(named_params, lr, weight_decay)
        self.betas, self.rho, self.hessian_every = betas, rho, hessian_every

    def update_hessian(self, grads, batch_tokens):
        b2 = self.betas[1]
        for name, p in self.params:
            g = grads.get(name)
            if g is None:
                continue
            st = self._st(name)
            est = batch_tokens * g * g
            st["h"] = b2 * st["h"] + (1 - b2) * est if "h" in st else (1 - b2) * est

    def _update(self, name, p, g):
        st = self._st(name)
        b1 = self.betas[0]
        st["m"] = b1 * st["m"] + (1 - b1) * g if "m" in st else (1 - b1) * g
        h = st.get("h", xp.zeros_like(p.data))
        ratio = st["m"] / xp.maximum(self.rho * h, 1e-12)
        self._decay(name, p)
        p.data -= self.lr * xp.clip(ratio, -1.0, 1.0)


def inverse_root(M, root, damping=1e-4):
    """M^(-1/root) for a symmetric PSD matrix via eigendecomposition.
    Damping is relative to the largest eigenvalue for scale invariance."""
    w, V = xp.linalg.eigh(M)
    w = xp.maximum(w, 0.0)
    w = w + damping * w.max() + 1e-12
    return (V * w ** (-1.0 / root)) @ V.T


class Shampoo(Optimizer):
    """Shampoo for 2D hidden matrices:  P = L^(-1/4) M R^(-1/4),
    with L, R exponential averages of G G^T and G^T G, M the momentum.
    The step magnitude is grafted from Adam (per matrix), which keeps the
    LR on the AdamW scale. Preconditioners are recomputed every k steps."""

    def __init__(self, named_params, lr, weight_decay=0.0, betas=(0.9, 0.95), beta_pre=0.99,
                 precondition_every=10, damping=1e-4):
        super().__init__(named_params, lr, weight_decay)
        self.betas, self.beta_pre = betas, beta_pre
        self.k, self.damping = precondition_every, damping

    def _update(self, name, p, g):
        if not is_hidden_matrix(name, p):
            return _adamw(self, name, p, g, self.betas)
        st = self._st(name)
        b1, b2 = self.betas
        if "m" not in st:
            m, n = p.shape
            st.update(m=xp.zeros_like(p.data), v=xp.zeros_like(p.data), t=0,
                      L=xp.zeros((m, m), dtype=p.data.dtype), R=xp.zeros((n, n), dtype=p.data.dtype))
        st["t"] += 1
        st["m"] = b1 * st["m"] + (1 - b1) * g
        st["v"] = b2 * st["v"] + (1 - b2) * g * g
        bp = self.beta_pre
        st["L"] = bp * st["L"] + (1 - bp) * (g @ g.T)
        st["R"] = bp * st["R"] + (1 - bp) * (g.T @ g)
        if st["t"] == 1 or st["t"] % self.k == 0:
            st["Li"] = inverse_root(st["L"], 4, self.damping)
            st["Ri"] = inverse_root(st["R"], 4, self.damping)
        mh = st["m"] / (1 - b1 ** st["t"])
        vh = st["v"] / (1 - b2 ** st["t"])
        adam_dir = mh / (xp.sqrt(vh) + 1e-8)
        P = st["Li"] @ mh @ st["Ri"]
        P = P * (xp.sqrt((adam_dir * adam_dir).sum()) / (xp.sqrt((P * P).sum()) + 1e-12))
        self._decay(name, p)
        p.data -= self.lr * P


OPTIMIZERS = {
    "sgd": SGD,
    "adamw": AdamW,
    "lion": Lion,
    "muon": Muon,
    "normuon": NorMuon,
    "sophia": Sophia,
    "shampoo": Shampoo,
}


def build_optimizer(name, named_params, lr, weight_decay):
    return OPTIMIZERS[name](named_params, lr=lr, weight_decay=weight_decay)
