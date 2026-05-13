import numpy as np
from typing import Optional

class EMASmoother:
    """
    Exponential Moving Average smoother for scalar values.

    Implements the formula: S_t = alpha * X_t + (1 - alpha) * S_{t-1}
    """
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.current_value: Optional[float] = None

    def update(self, value: float) -> float:
        """Update the EMA with a new value and return the smoothed result."""
        if self.current_value is None:
            self.current_value = value
        else:
            self.current_value = self.alpha * value + (1 - self.alpha) * self.current_value
        return self.current_value

    def reset(self) -> None:
        """Reset the smoother state."""
        self.current_value = None
