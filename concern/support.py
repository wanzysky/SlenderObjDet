from typing import List
from collections import Iterable

import numpy as np
import cv2
import matplotlib.pyplot as plt


def any_of(array, function, *args, **kwargs):
    if not isinstance(array, Iterable):
        return function(array, *args, **kwargs)
    for item in array:
        if function(*args, item, **kwargs):
            return True
    return False


def between(a, a_range: tuple) -> bool:
    if isinstance(a, np.ndarray):
        return np.logical_and(a >= a_range[0], a <= a_range[1])
    return a >= a_range[0] and a <= a_range[1]


def between_ranges(a, ranges: List[tuple]) -> List[bool]:
    return [between(a, a_range) for a_range in ranges]


def all_the_same(a_list: list) -> bool:
    for item in a_list:
        if not item == a_list[0]:
            return False
    return True


def make_dual(item_or_tuple) -> tuple:
    if isinstance(item_or_tuple, tuple):
        return item_or_tuple
    return (item_or_tuple, item_or_tuple)


def ratio_of_bbox(bbox):
    """
    Args:
        bbox: box in form (x0, y0, x1, y1).
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    if w * h == 0:
        return 0.0

    return min(w, h) / max(w, h)


def ratio_of_polygon(polygon):
    """
    Args:
        polygon (n, 2): a set of points.
    """

    polygon = np.concatenate(polygon, 0).reshape(-1, 2)
    hull = cv2.convexHull(polygon.astype(np.float32)).reshape(-1, 2)
    if hull.shape[0] < 3:
        return ratio_of_bbox([
            polygon[:, 0].min(),
            polygon[:, 1].min(),
            polygon[:, 0].max(),
            polygon[:, 1].max()
        ])
    rect = cv2.minAreaRect(hull.astype(np.float32))
    w, h = rect[1]
    if w * h == 0:
        return 0.0
    return min(w, h) / max(w, h)


def rbox_from_polygon(polygon) -> list:
    polygon = np.concatenate(polygon, 0).reshape(-1, 2)
    hull = cv2.convexHull(polygon.astype(np.float32)).reshape(-1, 2)
    center, size, angle = cv2.minAreaRect(hull.astype(np.float32))
    angle = -angle
    while angle < 0:
        angle += 180
    while angle > 180:
        angle -= 180

    if angle > 90:
        size = [size[1], size[0]]
        angle = angle - 90
    if angle > 45:
        angle -= 90
        size = [size[1], size[0]]
    return [*center, *size, angle]


def fig2image(fig: plt.Figure):
    fig.canvas.draw()
    buff, (width, height) = fig.canvas.print_to_buffer()
    image = np.fromstring(buff, dtype=np.uint8).reshape(height, width, 4)
    return image

def bounding_of_rbox(box):
    box = cv2.boundingRect(cv2.boxPoints(((box[0], box[1]), (box[2], box[3]), -box[4])))
    return np.array(box).tolist()
