import numpy as np
from typing import TypeAlias
from numba import njit  # type: ignore
from numba.types import uint8  # type: ignore

ColorRGB: TypeAlias = np.ndarray

@njit(uint8[::1](uint8, uint8, uint8), inline="always")
def Color(red: int, green: int, blue: int) -> ColorRGB:
    return np.array([red, green, blue], dtype=np.uint8)

@njit(uint8[::1](), inline="always")
def Black() -> ColorRGB:
    return Color(0, 0, 0)