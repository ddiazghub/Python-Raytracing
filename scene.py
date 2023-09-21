import numpy as np
from numpy import random
from camera import Camera, CameraType
from typing import TypeAlias
from color import Black, ColorRGB, White, blend
from vector import Point3D, Vector3, dot, norm2, normalize
from ray import ChildRays, NullRay, Ray
from numba.experimental import jitclass  # type: ignore
from numba.typed import typedlist  # type: ignore
from numba import types, njit, typed # type: ignore
from object import Floor, FloorType, LightSource, LightSourceType, Material, NullSphere, Sphere, SphereType

Image: TypeAlias = np.ndarray

@jitclass(spec=[("objects", types.ListType(SphereType)), ("camera", CameraType), ("floor", FloorType), ("lightsource", LightSourceType), ("background", types.uint8[::1])])
class Scene:
    """A 3D scene that can be rendered in a 2D image."""
    objects: typedlist.List[Sphere]
    camera: Camera
    background: ColorRGB
    lightsource: LightSource
    floor: Floor
    
    def __init__(self, camera: Camera, lightsource: LightSource, background: ColorRGB, floor_height: int, floor_material: Material) -> None:
        """A 3D scene that can be rendered in a 2D image.

        Args:
            camera (Camera): The scene's camera.
            lightsource (LightSource): The scene's lightsource's.
            background (ColorRGB): Background color.
            floor_height (int): The floor's y axis position.
            floor_material (Material): The floor's material.
        """
        self.objects = typedlist.List.empty_list(SphereType)
        self.camera = camera
        self.lightsource = lightsource
        self.background = background
        self.floor = Floor(floor_height, floor_material)
    
    def add(self, obj: Sphere) -> None:
        """Adds an object to the scene.

        Args:
            obj (Sphere): Object to add.
        """
        print("Added object", obj)
        self.objects.append(obj)
    
    def render(self, bitmap: Image, samples: int):
        """Renders the scene in a bitmap.

        Args:
            bitmap (Image): The bitmap where the scene will be rendered.
            samples (int): Number of samples per pixel to take.
        """
        print("Rendering scene")
        lower_left, origin = self.camera.viewport.lower_left, self.camera.position
        image_height, image_width, _ = bitmap.shape
        image_height -= 1
        width_ratio, height_ratio = self.camera.viewport.width / image_width, self.camera.viewport.height / image_height
        black = Black().astype(np.int32)
        rand = random.rand(samples, 2)
        
        for i in np.arange(image_height, -1, -1):
            for j in np.arange(image_width):
                color = black.copy()

                for i_rand, j_rand in rand:
                    position = lower_left + Vector3((j + j_rand) * width_ratio, (i + i_rand) * height_ratio, 0)
                    direction = normalize(position - origin)
                    ray = Ray(origin, direction)
                    color += self.trace(ray, 20)

                bitmap[image_height - i, j] = (color / samples).astype(np.uint8)

    def trace(self, ray: Ray, max_depth: int) -> ColorRGB:
        return trace(self, ray, max_depth)

    def trace_sky(self, ray: Ray) -> ColorRGB:
        light_direction = normalize(self.lightsource.center - ray.origin)
        light_intensity = max(0, dot(ray.direction, light_direction))
        
        return self.lightsource.blend_with(self.background, light_intensity).astype(np.uint8)

    def intersects(self, ray: Ray) -> tuple[Sphere, Ray]:
        """Checks if a ray collides with any object in the scene.

        Args:
            ray (Ray): The ray.

        Returns:
            tuple[Sphere, Ray, bool]: Tuple containing the closest object that the ray collides with, a ray pointing outwards from the collision point and a boolean which is true when the ray collides from inside the object.
        """
        closest: tuple[Sphere, Ray] = (NullSphere(), NullRay())
        min_distance = np.iinfo(np.int32).max

        for obj in self.objects:
            intersect = obj.intersects(ray)
            
            if not intersect.is_null():
                dist = norm2(intersect.origin - self.camera.position)

                if dist < min_distance:
                    closest = (obj, intersect)
                    min_distance = dist

        return closest
    
    def shadow(self, ray: Ray) -> float:
        """Finds the strength of a shadow on an object.

        Args:
            light_ray (Ray): Ray pointing to the scenes's lightsource.

        Returns:
            float: Strength of the shadow.
        """
        shadow = 0

        for obj in self.objects:
            if not obj.intersects(ray).is_null():
                shadow += 1 - obj.shadow()

        return min(shadow, 1)


@njit
def trace(self: Scene, ray: Ray, depth: int) -> ColorRGB:
    """Traces a ray in order to find the pixel's color.

    Args:
        ray (Ray): The ray to trace.
        depth (int): Current ray reflection recursion depth.

    Returns:
        ColorRGB: The pixel's color.
    """
    #print("Recursion depth:", depth)
    obj, normal = self.intersects(ray)
    children: ChildRays
    facing_ratio: float

    if obj.is_null():
        normal = self.floor.intersects(ray)

        if normal.is_null():
            #print("Propagates to the sky")
            return self.trace_sky(ray)

        #print("Intersects floor")
        facing_ratio = dot(ray.direction, normal.direction)
        children = child_rays(self, ray, self.floor, normal, facing_ratio, depth)
        obj.material = self.floor.material
    else:
        #print("Intersects sphere with color:", obj.material.color)
        facing_ratio = dot(ray.direction, normal.direction)
        children = child_rays(self, ray, obj, normal, facing_ratio, depth)

    light_intensity = np.abs(dot(normal.direction, children.light.direction))
    intensity10 = light_intensity ** 10
    #print("Light intensity is: ", light_intensity)
    no_reflect = children.reflect.is_null()
    no_refract = children.refract.is_null()

    #print("No reflect: ", no_reflect)
    #print("No refract: ", no_refract)

    if no_reflect and no_refract:
        light_intensity *= 1 - self.shadow(children.light)
        diffuse_color = self.lightsource.blend_with(obj.material.color, intensity10)
        diffuse_color = (diffuse_color * light_intensity).astype(np.uint8)
        #print("Diffuse object. Pixel color:", diffuse_color)

        return diffuse_color

    #print("Tag1")
    #print("Reflection ray:", children.reflect.origin, children.reflect.direction)
    #print("Refraction ray:", children.refract.origin, children.refract.direction)
    #print("Material:", obj.material.color, obj.material.reflection, obj.material.refraction_index, obj.material.transparency)
    
    if no_reflect or no_refract:
        blend_ratio: float
        child: Ray

        if no_refract:
            blend_ratio = obj.material.reflection
            child = children.reflect
        else:
            if facing_ratio > 0:
                light_intensity = 1
                intensity10 = 1

            child = children.refract
            blend_ratio = obj.material.transparency

        blend_ratio *= blend_ratio * blend_ratio
        child_color = trace(self, child, depth - 1)
        light_intensity = (light_intensity * (1 - blend_ratio)) + blend_ratio
        pixel_color = blend(child_color, obj.material.color, blend_ratio)
        pixel_color = self.lightsource.blend_with(pixel_color, intensity10)
        #print("Reflective | refractive object. Pixel color:", pixel_color)

        return (pixel_color * light_intensity).astype(np.uint8)

    refraction_color = obj.material.color
    reflection_color: ColorRGB

    if children.fresnel < 1:
        refraction_color = trace(self, children.refract, depth - 1)

    #print("Facing ratio:", facing_ratio)
    #print("Reflective and refractive object.")
    reflection_color = trace(self, children.reflect, depth - 1) if facing_ratio < 0 else refraction_color
    #print("Reflection color:", reflection_color)
    #print("Refraction color:", refraction_color)
    pixel_color = reflection_color * children.fresnel + refraction_color * (1 - children.fresnel) * obj.material.transparency
    #print("pixel color 1:", pixel_color)
    pixel_color = (pixel_color * obj.material.color / White()).astype(np.uint8)
    #print("pixel color 2:", pixel_color)

    return pixel_color

    #if obj.material.reflection > 0:
    #    blend_ratio = obj.material.reflection
    #else:
    #    if facing_ratio > 0:
    #        light_intensity = 1
    #        intensity10 = 1
    #
    #   blend_ratio = obj.material.transparency
    #
    #blend_ratio *= blend_ratio * blend_ratio
    #child_color = trace(self, children.child, depth - 1)
    #light_intensity = (light_intensity * (1 - blend_ratio)) + blend_ratio
    #pixel_color = blend(child_color, obj.material.color, blend_ratio)
    #pixel_color = self.lightsource.blend_with(pixel_color, intensity10)

    #return (pixel_color * light_intensity).astype(np.uint8)

@njit(inline="always")
def child_rays(scene: Scene, ray: Ray, obj: Sphere | Floor, normal_ray: Ray, facing_ratio: float, depth: int) -> ChildRays:
    """Calculates and child rays after parent ray collision depending on object material type.

    Args:
        scene (Scene): The scene where the ray will be traced.
        obj (Sphere | Floor): Collision object.
        normal_ray (Ray): Ray pointing outwards from the collision point.

    Returns:
        Ray: Child ray to trace next.
    """
    #print("Ray:", ray.origin, ray.direction)
    #print("Normal ray:", normal_ray.origin, normal_ray.direction)
    light_direction = normalize(scene.lightsource.center - normal_ray.origin)
    light_ray = Ray(normal_ray.origin, light_direction)
    #print("Light ray:", light_ray.origin, light_direction)

    # Object is reflective/refractive
    if depth > 0:
        if obj.material.reflection > 0 and obj.material.transparency > 0:
            #print("Obj material:", obj.material.color, obj.material.reflection, obj.material.transparency, obj.material.refraction_index)
            reflection = reflect(ray, normal_ray, facing_ratio)
            refraction, kr = refract(ray, normal_ray, facing_ratio, obj.material.refraction_index)

            return ChildRays(reflection, refraction, light_ray, kr)

        if obj.material.reflection > 0:
            return ChildRays(reflect(ray, normal_ray, facing_ratio), NullRay(), light_ray, np.NaN)

        if obj.material.transparency > 0:
            #print("Obj material:", obj.material.color, obj.material.reflection, obj.material.transparency, obj.material.refraction_index)
            refraction, kr = refract(ray, normal_ray, facing_ratio, obj.material.refraction_index)

            return ChildRays(NullRay(), refraction, light_ray, kr)

    return ChildRays(NullRay(), NullRay(), light_ray, np.NaN)

@njit(inline="always")
def reflect(ray: Ray, normal_ray: Ray, facing_ratio: float) -> Ray:
    #print("Computing reflection")
    reflection_direction = ray.direction - 2 * facing_ratio * normal_ray.direction
    
    return Ray(normal_ray.origin, reflection_direction)

@njit(inline="always")
def refract(ray: Ray, normal_ray: Ray, facing_ratio: float, eta_ratio: float) -> tuple[Ray, float]:
    #print("Computing refraction")
    normal = normal_ray.direction
    cosine = facing_ratio
    #print("Facing ratio is", facing_ratio)
    #print("Eta ratio is", eta_ratio)

    if facing_ratio < 0:
        eta_ratio = 1 / eta_ratio
        cosine = -cosine
    else:
        normal = -normal
        
    #print("Inverted facing ratio")
    #print("Facing ratio:", facing_ratio)
    #print("Cosine:", cosine)
    #print("Eta ratio:", eta_ratio)
    k = 1 - (eta_ratio * eta_ratio) * (1 - cosine * cosine)
    refraction_direction = (ray.direction + facing_ratio * normal) * eta_ratio - normal * np.sqrt(k)
    fresnel_effect = fresnel(facing_ratio, eta_ratio)
    #print("Computed fresnel")
    refraction_ray = Ray(normal_ray.origin, refraction_direction)
    refraction_ray.origin = refraction_ray.propagate(0.001)
    #print("Found refraction ray")
    return (refraction_ray, fresnel_effect)

@njit
def fresnel(facing_ratio: float, eta_ratio: float) -> float:
    #print("Computing fresnel")
    sin_t = eta_ratio * np.sqrt(max(1 - facing_ratio * facing_ratio, 0))

    if sin_t < 1:
        cos_t = np.sqrt(max(1 - sin_t * sin_t, 0))
        cos_i = np.abs(cos_t)
        eta_i, eta_t = (eta_ratio, 1) if eta_ratio > 0 else (1, 1 / eta_ratio)
        etat_cosi, etai_cost = eta_t * cos_i, eta_i * cos_t
        etai_cosi, etat_cost = eta_i * cos_i, eta_t * cos_t
        r_s = (etat_cosi - etai_cost) / (etat_cosi + etai_cost)
        r_p = (etai_cosi - etat_cost) / (etai_cosi + etat_cost)

        return (r_s * r_s + r_p * r_p) / 2
        
    # Total internal reflection
    return 1