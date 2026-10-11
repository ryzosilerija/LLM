"""Learning-rate schedules. All return a multiplier in [0, 1] for base_lr.

    constant  linear warmup, then flat
    cosine    linear warmup, then cosine decay to min_ratio
    wsd       warmup-stable-decay: warmup, flat, linear decay over the
              last `decay_frac` of training to min_ratio
"""
import math


def lr_multiplier(step, max_steps, warmup, kind="cosine", min_ratio=0.1, decay_frac=0.2):
    if warmup and step < warmup:
        return (step + 1) / warmup
    if kind == "constant":
        return 1.0
    if kind == "cosine":
        progress = (step - warmup) / max(1, max_steps - warmup)
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))
    if kind == "wsd":
        decay_start = int(max_steps * (1 - decay_frac))
        if step < decay_start:
            return 1.0
        progress = (step - decay_start) / max(1, max_steps - decay_start)
        return 1.0 - (1 - min_ratio) * min(1.0, progress)
    raise ValueError(f"unknown schedule {kind!r}")
