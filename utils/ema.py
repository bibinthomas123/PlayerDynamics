import numpy as np
from typing import Optional

class EMASmoother:
    """
    Exponential Moving Average smoother for scalar values.

    Implements the formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float):
        self._alpha = alpha
        self.current_value: Optional[float] = None
        self._ema = None

    def update(self, value: float) -> float:
        if self._ema is None:
            self._ema = value
        else:
            self._ema = self._alpha * value + (1 - self._alpha) * self._ema
        return self._ema

    @property
    def value(self) -> float:
        """Last EMA value without triggering an update. Returns 0.0 if never updated."""
        return self._ema if self._ema is not None else 0.0

    def reset(self) -> None:
        """Reset the smoother state."""
        self.current_value = None
