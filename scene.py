import numpy as np
from numpy import random
from numpy.random.mtrand import normal
from camera import Camera, CameraType
from typing import TypeAlias
from color import Black, ColorRGB, blend
from vector import Point3D, Vector3, dot, norm2, normalize
from ray import ChildRays, NullRay, Ray
from numba.experimental import jitclass  # type: ignore
from numba.typed import typedlist  # type: ignore
from numba import types, njit, typed # type: ignore
from object import Floor, FloorType, LightSource, LightSourceType, Material, MaterialType, NullSphere, Propagation, PropagationType, Sphere, SphereType

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
                    color += self.trace(ray, 5)

                bitmap[image_height - i, j] = (color / samples).astype(np.uint8)

    def trace(self, ray: Ray, max_depth: int) -> ColorRGB:
        """Traces a ray in order to find the pixel's color.
        (This was supposed to be a recursive function but for some reason numba doesn't support recursion).

        Args:
            ray (Ray): The ray to trace.
            depth (int): Current ray reflection recursion depth.

        Returns:
            ColorRGB: The pixel's color.
        
        depth = max_depth
        ray = starting_ray
        color = self.background
        propagation_stack: list[Propagation] = typedlist.List.empty_list(PropagationType)

        while True:
            #print("Recursion depth:", depth)
            obj, normal, inside = self.intersects(ray)
            children: ChildRays

            if obj.is_null():
                #print("Intersects floor")
                normal = self.floor.intersects(ray)

                if normal.is_null():
                    color = self.trace_sky(ray)
                    break
                
                children = child_rays(self, ray, self.floor, normal, depth)
                obj.material = self.floor.material
            else:
                #print("Intersects sphere with color:", obj.material.color)
                children = child_rays(self, ray, obj, normal, depth)
    
            # I'm really sorry. Numba does not support match statements
            case = children.get_type()

            if case == 0:
                #print("Diffuse obj In shadow")
                color = Black()
                break

            light_intensity = max(0, dot(normal.direction, children.light.direction))

            if case == 1:
                diffuse_color = self.lightsource.blend_with(obj.material.color, light_intensity ** 10)
                color = (diffuse_color * light_intensity).astype(np.uint8)
                #print("Diffuse object. Pixel color:", color)
                break

            propagation_stack.append(Propagation(obj.material, self.floor.material, light_intensity))
            #print("Reflective object. Material:", obj.material.color, obj.material.reflection, obj.material.transparency)
            ray = children.reflect
            depth -= 1

        color = self.unwind_stack(propagation_stack, color)

        return color
        """

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

    if obj.is_null():
        normal = self.floor.intersects(ray)

        if normal.is_null():
            #print("Propagates to the sky")
            return self.trace_sky(ray)
        
        #print("Intersects floor")
        children = child_rays(self, ray, self.floor, normal, depth)
        obj.material = self.floor.material
    else:
        #print("Intersects sphere with color:", obj.material.color)
        children = child_rays(self, ray, obj, normal, depth)

    light_intensity = np.abs(dot(normal.direction, children.light.direction))
    intensity10 = light_intensity ** 10
    #print("Light intensity is: ", light_intensity)

    if children.reflect.is_null():
        light_intensity *= 1 - self.shadow(children.light)
        diffuse_color = self.lightsource.blend_with(obj.material.color, intensity10)
        diffuse_color = (diffuse_color * light_intensity).astype(np.uint8)
        #print("Diffuse object. Pixel color:", diffuse_color)

        return diffuse_color

    refraction_color = Black()
    reflection_color = Black()
    #facing_ratio = dot(normal.direction, ray.direction)
    #inverse_ratio = 1 - facing_ratio
    #fresnel = blend(inverse_ratio * inverse_ratio * inverse_ratio, 1, 0.1)
    #reflect_blend = 0
    #refract_blend = 0
    #print("Reflective/Refractive object")

    #if obj.material.reflection > 0 and obj.material.transparency > 0:
        #reflect_blend, refract_blend = fresnel, 1 - fresnel

    if obj.material.reflection > 0:
        #print("Ray reflected by:", propagation.material.color, "Reflection color:", color)
        reflection_color = trace(self, children.reflect, depth - 1)
        #print("Reflection color:", reflection_color)
        #if refract_blend == 0:
            #reflect_blend = 1
        
    if obj.material.transparency > 0:
        refraction_color = trace(self, children.refract, depth - 1)
        #print("Refraction color:", refraction_color)
        #if reflect_blend == 0:
            #refract_blend = 1

    intensity_index = max(obj.material.transparency, obj.material.transparency)
    light_intensity = (light_intensity * (1 - intensity_index)) + intensity_index 
    pixel_color = blend(reflection_color, blend(refraction_color, obj.material.color, obj.material.transparency), obj.material.reflection)
    pixel_color = self.lightsource.blend_with(pixel_color, intensity10)
    #print("Pixel color:", pixel_color * light_intensity)

    return (pixel_color * light_intensity).astype(np.uint8)

@njit(inline="always")
def child_rays(scene: Scene, ray: Ray, obj: Sphere | Floor, normal_ray: Ray, depth: int) -> ChildRays:
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
    if (obj.material.reflection > 0 or obj.material.transparency > 0) and depth > 0:
        facing_ratio = dot(ray.direction, normal_ray.direction)
        reflection_direction = ray.direction - 2 * facing_ratio * normal_ray.direction
        reflection_ray = Ray(normal_ray.origin, reflection_direction)
        refraction_ray = refract(ray, normal_ray, facing_ratio, obj.material.refraction_index)
        #print("Object is reflective/refractive. Reflection ray:", normal_ray.origin, reflection_direction, "Refraction ray:", refraction_ray.origin, refraction_ray.direction)
        
        return ChildRays(reflection_ray, refraction_ray, light_ray)

    return ChildRays(NullRay(), NullRay(), light_ray)

@njit(inline="always")
def refract(ray: Ray, normal_ray: Ray, facing_ratio: float, eta_ratio: float) -> Ray:
    normal = normal_ray.direction
    cosine = facing_ratio

    if facing_ratio > 0:
        normal = -normal
    else:
        eta_ratio = 1 / eta_ratio
        cosine = -cosine
    
    #print("Facing ratio:", facing_ratio)
    #print("Cosine:", cosine)
    #print("Eta ratio:", eta_ratio)
    perpendicular = eta_ratio * (ray.direction + cosine * normal_ray.direction)
    parallel = -np.sqrt(np.abs(1 - norm2(perpendicular))) * normal_ray.direction

    return Ray(normal_ray.origin - normal_ray.direction * 0.001, perpendicular + parallel)