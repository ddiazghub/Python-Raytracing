import numpy as np
from numba.experimental import jitclass  # type: ignore
from numba.types import double, uint8  # type: ignore
from numba import njit
from numpy.matrixlib.defmatrix import mat  # type: ignore
from vector import Point3D, Origin, Vector3, dot, norm2, normalize
from color import Black, ColorRGB, blend
from ray import NullRay, Ray

@jitclass(spec=[("color", uint8[::1]), ("transparency", double), ("reflection", double)])
class Material:
    color: ColorRGB
    transparency: float
    reflection: float

    def __init__(self, color: ColorRGB, transparency: float, reflection: float) -> None:
        self.color = color
        self.transparency = transparency
        self.reflection = reflection

MaterialType = Material.class_type.instance_type # type: ignore

@jitclass(spec=[("height", double), ("material", MaterialType)])
class Floor:
    height: float
    material: Material

    def __init__(self, height: float, material: Material) -> None:
        self.height = height
        self.material = material
    
    def intersects(self, ray: Ray) -> Ray:
        """
        Finds if a ray is intersected by the floor.
        Floor intersects ray if (ray.origin.y < floor.height || ray.direction.y < 0).
        Point of intersection is when ray.origin.y + time * ray.direction.y == floor.height
        """
        if ray.origin[1] < self.height:
            return Ray(ray.origin, Vector3(0, 0, -1))

        time = (self.height - ray.origin[1]) / ray.direction[1]

        if time < 0:
            return NullRay()
        
        collision_point = ray.propagate(time)
        normal = Vector3(0, 1, 0)

        return Ray(collision_point, normal)

@jitclass(spec=[("center", double[::1]), ("radius", double), ("material", MaterialType)])
class Sphere:
    center: Point3D
    radius: float
    material: Material

    def __init__(self, center: Point3D, radius: float, material: Material) -> None:
        self.center = center
        self.radius = radius
        self.material = material
    
    def intersects(self, ray: Ray) -> tuple[Ray, bool]:
        return sphere_intersects(self, ray)

    def is_null(self) -> bool:
        return self.radius == 0

@jitclass(spec=[("center", double[::1]), ("radius", double), ("color", uint8[::1])])
class LightSource:
    center: Point3D
    radius: float
    color: ColorRGB

    def __init__(self, center: Point3D, radius: float, color: ColorRGB) -> None:
        self.center = center
        self.radius = radius
        self.color = color
    
    def blend_with(self, color: ColorRGB, intensity: float) -> ColorRGB:
        return blend(self.color, color, intensity)

@jitclass(spec=[("material", MaterialType), ("attenuation", double)])
class Propagation:
    material: Material
    light_intensity: float

    def __init__(self, material: Material, attenuation: float) -> None:
        self.material = material
        self.light_intensity = attenuation

@njit
def sphere_intersects(self: Sphere | LightSource, ray: Ray) -> tuple[Ray, bool]:
    """
    Sphere intersects ray if (ray.origin + t * ray.direction − sphere.center)^2 <= sphere.radius^2 for any t.
    We solve the quadratic formula to find time of intersection.
    """

    origin_center = ray.origin - self.center
    a = norm2(ray.direction)
    half_b = dot(ray.direction, origin_center)
    c = norm2(origin_center) - self.radius * self.radius
    discriminant = half_b * half_b - a * c

    if discriminant > 0:
        disc_sqrt = np.sqrt(discriminant)
        collision_t = (-half_b - disc_sqrt) / a
        inside = False

        if collision_t < 0.0001:
            collision_t = (-half_b + disc_sqrt) / a

            if collision_t < 0.0001:
                return (NullRay(), False)

        collision_point = ray.propagate(collision_t)
        normal = (collision_point - self.center) / self.radius

        return (Ray(collision_point, -normal if inside else normal), inside)
    
    return (NullRay(), False)

@njit(inline="always")
def NullSphere() -> Sphere:
    return Sphere(Origin(), 0, Material(Black(), 0, 0))

FloorType = Floor.class_type.instance_type # type: ignore
SphereType = Sphere.class_type.instance_type # type: ignore
LightSourceType = LightSource.class_type.instance_type # type: ignore
PropagationType = Propagation.class_type.instance_type # type: ignore