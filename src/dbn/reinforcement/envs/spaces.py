"""Minimal Gymnasium-like spaces (no gym dependency)."""

from __future__ import annotations

import numpy as np


class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            low_a = np.asarray(low, dtype=dtype)
            high_a = np.asarray(high, dtype=dtype)
            shape = low_a.shape
            self.low = low_a
            self.high = high_a
        else:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype

    def sample(self, rng: np.random.Generator | None = None):
        rng = rng or np.random.default_rng()
        return rng.uniform(self.low, self.high).astype(self.dtype)


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64

    def sample(self, rng: np.random.Generator | None = None) -> int:
        rng = rng or np.random.default_rng()
        return int(rng.integers(0, self.n))
