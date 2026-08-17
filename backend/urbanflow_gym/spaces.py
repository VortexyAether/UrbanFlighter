from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BoxSpec:
    """Small NumPy-only subset of the Gymnasium ``Box`` contract."""

    low: np.ndarray
    high: np.ndarray
    dtype: np.dtype = np.dtype(np.float32)

    def __post_init__(self) -> None:
        low = np.asarray(self.low, dtype=self.dtype)
        high = np.asarray(self.high, dtype=self.dtype)
        if low.shape != high.shape:
            raise ValueError("BoxSpec low/high shapes must match")
        if np.any(low > high):
            raise ValueError("BoxSpec low values must not exceed high values")
        object.__setattr__(self, "low", low)
        object.__setattr__(self, "high", high)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.low.shape

    def contains(self, value: object) -> bool:
        candidate = np.asarray(value)
        return bool(
            candidate.shape == self.shape
            and np.all(np.isfinite(candidate))
            and np.all(candidate >= self.low)
            and np.all(candidate <= self.high)
        )

    def to_dict(self) -> dict:
        return {
            "type": "Box",
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "provider": "urbanflow_gym.numpy",
        }
