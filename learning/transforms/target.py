import numpy as np

from PIL import Image
from skimage.morphology import opening, closing, convex_hull_image
from typing import Callable


def transform(transform_fn: Callable, image: Image, mult: int = 1, **kwargs):
    img = np.array(image)
    img = mult * transform_fn(img, **kwargs).astype(np.uint8)
    img = Image.fromarray(img)

    return img.point(lambda p: p > 255 // 2 and 255)


def bounding_box(img):
    min_x, max_x, min_y, max_y = np.inf, 0, np.inf, 0

    for i, row in enumerate(img):
        for j, val in enumerate(row):
            if val > 250:
                if i < min_y:
                    min_y = i
                if i > max_y:
                    max_y = i

                if j < min_x:
                    min_x = j
                if j > max_x:
                    max_x = j

    if min_x == np.inf:
        min_x = 0
    if min_y == np.inf:
        min_y = 0

    new_img = np.zeros_like(img)
    bounding_box = np.ones((max_y - min_y, max_x - min_x))
    new_img[min_y:max_y, min_x:max_x] = bounding_box

    return new_img


class Opening:
    def __init__(self, selem_fn: Callable, factor: int):
        self.selem = selem_fn(factor)

    def __call__(self, img: Image):
        return transform(opening, img, selem=self.selem)


class Closing:
    def __init__(self, selem_fn: Callable, factor: int):
        self.selem = selem_fn(factor)

    def __call__(self, img: Image):
        return transform(closing, img, selem=self.selem)


class ConvexHull:
    def __call__(self, img: Image):
        return transform(convex_hull_image, img, mult=255)


class BoundingBox:
    def __call__(self, img: Image):
        return transform(bounding_box, img)
