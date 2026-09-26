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
        self._rng = np.random.default_rng()

    def sample(self, rng: np.random.Generator | None = None, n: int | None = None):
        """Sample one action, or a batch of ``n`` actions with shape (n, *shape)."""
        rng = rng or self._rng
        if n is None:
            return rng.uniform(self.low, self.high).astype(self.dtype)
        size = (int(n),) + self.shape
        return rng.uniform(self.low, self.high, size=size).astype(self.dtype)


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64
        self._rng = np.random.default_rng()

    def sample(self, rng: np.random.Generator | None = None, n: int | None = None):
        rng = rng or self._rng
        if n is None:
            return int(rng.integers(0, self.n))
        return rng.integers(0, self.n, size=int(n), dtype=self.dtype)
