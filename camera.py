from vector import Point3D, Vector3
from numba import types  # type: ignore
from numba.experimental import jitclass # type: ignore

@jitclass(spec=[("position", types.double[::1]), ("width", types.double), ("height", types.double), ("aspect_ratio", types.double), ("lower_left", types.double[::1])])
class Viewport:
    position: Point3D
    width: float
    height: float
    aspect_ratio: float
    lower_left: Point3D

    def __init__(self, position: Point3D, width: float, height: float) -> None:
        self.position = position
        self.width = width
        self.height = height
        self.aspect_ratio = width / height
        self.lower_left = Vector3(position[0] - width / 2, position[1] - height / 2, position[2])

ViewportType = Viewport.class_type.instance_type # type: ignore

@jitclass(spec=[("position", types.double[::1]), ("viewport", ViewportType)])
class Camera:
    position: Point3D
    viewport: Viewport

    def __init__(self, position: Point3D, viewport_width: float, viewport_height: float, focal_length: float) -> None:
        self.position = position
        self.viewport = Viewport(position + Vector3(0, 0, focal_length), viewport_width, viewport_height)

CameraType = Camera.class_type.instance_type # type: ignore