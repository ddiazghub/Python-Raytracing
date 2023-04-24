import numpy as np
from numba import njit  # type: ignore
from numba.experimental import jitclass  # type: ignore
from numba.types import double  # type: ignore
from vector import Point3D, Origin, Vector3D, VectorZero

@jitclass(spec=[("origin", double[::1]), ("direction", double[::1])])
class Ray:
    origin: Point3D
    direction: Vector3D

    def __init__(self, origin: Point3D, direction: Vector3D) -> None:
        self.origin = origin
        self.direction = direction
    
    def propagate(self, t: int) -> Point3D:
        return self.origin + t * self.direction

    def is_null(self) -> bool:
        return np.array_equal(self.direction, VectorZero())

RayType = Ray.class_type.instance_type # type: ignore

@njit(inline="always")
def NullRay() -> Ray:
    return Ray(Origin(), VectorZero())