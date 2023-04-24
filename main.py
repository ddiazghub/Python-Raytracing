import numpy as np
from color import Color
from object import Material, Sphere
from scene import Scene
from camera import Camera
from vector import Origin, Point3
from PIL import Image

"""
WIDTH = 256
HEIGHT = 256
BLUE = 255 // 4

def image_vectorization() -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), 255, dtype=np.uint8)
    image[:, :, 2] = BLUE
    image[:, :, :2] -= np.indices((HEIGHT, WIDTH), dtype=np.uint8).transpose(1, 2, 0)

    return image
"""

IMAGE_WIDTH = 400
IMAGE_HEIGHT = 200
VIEWPORT_WIDTH = 4
VIEWPORT_HEIGHT = 2
FOCAL_DISTANCE = 1
LIGHTSOURCE_POSITION = Point3(0, 10, -5)
BACKGROUND = Color(173, 216, 230)
FLOOR = Material(Color(209, 206, 2), 0, 0)
RED = Color(255, 0, 0)
GREEN = Color(0, 255, 0)
BLUE = Color(0, 0, 255)
WHITE = Color(255, 255, 255)
SAMPLES = 5

if __name__ == "__main__":
    camera = Camera(Origin(), VIEWPORT_WIDTH, VIEWPORT_HEIGHT, FOCAL_DISTANCE)
    scene = Scene(camera, LIGHTSOURCE_POSITION, BACKGROUND, -1, FLOOR)
    scene.add(Sphere(Point3(0, 0, 3), 1, Material(RED, 0, 0)))
    scene.add(Sphere(Point3(2, 0, 3), 1, Material(GREEN, 0, 0.7)))
    #scene.add(Sphere(Point3(10, 0, 20), 13, BLUE))
    #scene.add(Sphere(Point3(-18, 0, 30), 20, WHITE))
    screen = np.empty((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    scene.render(screen, SAMPLES)
    Image.fromarray(screen).save("out.bmp")