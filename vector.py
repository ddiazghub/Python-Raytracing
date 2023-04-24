import numpy as np
from typing import TypeAlias
from numba import njit # type: ignore
from numba.types import double  # type: ignore

Vector3D: TypeAlias = np.ndarray
Point3D: TypeAlias = Vector3D

@njit(double[::1](double, double, double), inline="always")
def Vector3(x: float, y: float, z: float) -> Vector3D:
    return np.array([x, y, z], dtype=np.float64)

@njit(double[::1](), inline="always")
def VectorZero() -> Vector3D:
    return Vector3(0, 0, 0)

@njit(double[::1](double, double, double), inline="always")
def Point3(x: float, y: float, z: float) -> Point3D:
    return Vector3(x, y, z)

@njit(double[::1](), inline="always")
def Origin() -> Point3D:
    return Point3(0, 0, 0)

@njit(double(double[::1], double[::1]), inline="always")
def dot(self: Vector3D, other: Vector3D) -> float:
    return np.sum(self * other)

@njit(double(double[::1]), inline="always")
def norm2(self: Vector3D) -> float:
    return dot(self, self)

@njit(double(double[::1]), inline="always")
def norm(self: Vector3D) -> float:
    return np.sqrt(norm2(self))

@njit(double[::1](double[::1]), inline="always")
def normalize(self: Vector3D) -> Vector3D:
    return self / norm(self)