import numpy as np
from numpy import random
from camera import Camera, CameraType
from typing import TypeAlias
from color import Black, ColorRGB, blend
from vector import Point3D, Vector3, dot, norm2, normalize
from ray import NullRay, Ray
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
                    direction = position - origin
                    ray = Ray(origin, direction)
                    color += self.trace(ray, 5)

                bitmap[image_height - i, j] = (color / samples).astype(np.uint8)

    def trace(self, starting_ray: Ray, max_depth: int) -> ColorRGB:
        """Traces a ray in order to find the pixel's color.
        (This was supposed to be a recursive function but for some reason numba doesn't support recursion).

        Args:
            ray (Ray): The ray to trace.
            depth (int): Current ray reflection recursion depth.

        Returns:
            ColorRGB: The pixel's color.
        """
        depth = max_depth
        ray = starting_ray
        color = self.background
        propagation_stack: list[Propagation] = typedlist.List.empty_list(PropagationType)

        while True:
            #print("Recursion depth:", depth)
            obj, normal, inside = self.intersects(ray)
            child: Ray
            light: Ray

            if obj.is_null():
                #print("Intersects floor")
                normal = self.floor.intersects(ray)

                if normal.is_null():
                    color = self.trace_sky(ray)
                    break
                
                child, light = child_ray(self, ray, self.floor, normal, depth)
                obj.material = self.floor.material
            else:
                #print("Intersects sphere with color:", obj.material.color)
                child, light = child_ray(self, ray, obj, normal, depth)

            if light.is_null():
                #print("Diffuse obj In shadow")
                color = Black()
                break

            light_intensity = max(0, dot(normal.direction, light.direction))

            if child.is_null():
                diffuse_color = self.lightsource.blend_with(obj.material.color, light_intensity ** 10)
                color = (diffuse_color * light_intensity).astype(np.uint8)
                #print("Diffuse object. Pixel color:", color)
                break

            propagation_stack.append(Propagation(obj.material, light_intensity))
            #print("Reflective object. Material:", obj.material.color, obj.material.reflection, obj.material.transparency)
            ray = child
            depth -= 1

        color = self.unwind_stack(propagation_stack, color)

        return color

    def trace_sky(self, ray: Ray) -> ColorRGB:
        direction = normalize(ray.direction)
        light_direction = normalize(self.lightsource.center - ray.origin)
        light_intensity = max(0, dot(direction, light_direction))
        
        return self.lightsource.blend_with(self.background, light_intensity).astype(np.uint8)

    def unwind_stack(self, propagation_stack: list[Propagation], final_color: ColorRGB) -> ColorRGB:
        color = final_color

        while len(propagation_stack) > 0:
            propagation = propagation_stack.pop()
            #print("Ray reflected by:", propagation.material.color, "Reflection color:", color)
            child_color = blend(color, propagation.material.color, propagation.material.reflection)
            child_color = self.lightsource.blend_with(child_color, propagation.light_intensity ** 10)
            color = (child_color * propagation.light_intensity).astype(np.uint8)
            
        return color

    def intersects(self, ray: Ray) -> tuple[Sphere, Ray, bool]:
        """Checks if a ray collides with any object in the scene.

        Args:
            ray (Ray): The ray.

        Returns:
            tuple[Sphere, Ray, bool]: Tuple containing the closest object that the ray collides with, a ray pointing outwards from the collision point and a boolean which is true when the ray collides from inside the object.
        """
        closest: tuple[Sphere, Ray, bool] = (NullSphere(), NullRay(), False)
        min_distance = np.iinfo(np.int32).max

        for obj in self.objects:
            intersect, inside = obj.intersects(ray)
            
            if not intersect.is_null():
                dist = norm2(intersect.origin - self.camera.position)

                if dist < min_distance:
                    closest = (obj, intersect, inside)
                    min_distance = dist

        return closest

    

@njit(inline="always")
def child_ray(scene: Scene, ray: Ray, obj: Sphere | Floor, normal_ray: Ray, depth: int) -> tuple[Ray, Ray]:
    """Calculates and traces child ray after parent ray collision depending on object material type.

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
        reflection_direction = ray.direction - 2 * dot(ray.direction, normal_ray.direction) * normal_ray.direction
        #print("Object is reflective. Reflection ray:", normal_ray.origin, reflection_direction)
        return (Ray(normal_ray.origin, reflection_direction), light_ray)

    # Object is diffuse
    if scene.intersects(light_ray)[0].is_null():
        #print("Object is diffuse")
        return (NullRay(), light_ray)
    
    # Object is shadowed
    #print("Object is shadowed")
    return (NullRay(), NullRay())