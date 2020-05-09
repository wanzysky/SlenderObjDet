from typing import Any, Iterator, List, Union
from functools import lru_cache
import pyclipper

import numpy as np
import cv2

from detectron2.structures.masks import PolygonMasks, polygon_area
from detectron2.structures.boxes import Boxes

from concern.support import make_dual
from concern import webcv2


@lru_cache()
def standard_linear(resolution=128):
    grid = (np.mgrid[0:resolution, 0:resolution] / resolution).astype(np.float32).sum(0)
    return (grid < 1) * grid


def coordinate_transform(standard: np.ndarray, p_o, p_x, p_y, out_shape):
    h, w = standard.shape[:2]
    source_points = np.array([[0, 0], [0, h], [w, 0]], dtype=np.float32)
    dest_points = np.array([p_o, p_y, p_x], dtype=np.float32)
    M = cv2.getAffineTransform(source_points, dest_points)
    out_w = dest_points[:, 0].max() - dest_points[:, 0].min()
    out_h = dest_points[:, 1].max() - dest_points[:, 1].min()
    return cv2.warpAffine(standard, M, out_shape)


def perspective_transform(standard:np.ndarray, poly, out_shape):
    h, w = standard.shape[:2]
    source_points = np.array([[0, 0]])


def dilate_polygon(polygon: np.ndarray, distance):
    subject = [tuple(l) for l in polygon]
    padding = pyclipper.PyclipperOffset()
    padding.AddPath(subject, pyclipper.JT_ROUND,
		    pyclipper.ET_CLOSEDPOLYGON)
    return np.array(padding.Execute(distance)[0])


def draw_border_map(polygon, canvas, ratio):
    polygon = np.array(polygon)
    assert polygon.ndim == 2
    assert polygon.shape[1] == 2

    distance = np.sqrt(polygon_area(polygon[:, 0], polygon[:, 1])) * ratio
    padded_polygon = dilate_polygon(polygon, distance)

    xmin = padded_polygon[:, 0].min()
    xmax = padded_polygon[:, 0].max()
    ymin = padded_polygon[:, 1].min()
    ymax = padded_polygon[:, 1].max()
    width = xmax - xmin + 1
    height = ymax - ymin + 1

    polygon[:, 0] = polygon[:, 0] - xmin
    polygon[:, 1] = polygon[:, 1] - ymin

    xs = np.broadcast_to(
	np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
    ys = np.broadcast_to(
	np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

    distance_map = np.zeros(
	(polygon.shape[0], height, width), dtype=np.float32)
    for i in range(polygon.shape[0]):
        j = (i + 1) % polygon.shape[0]
        absolute_distance = compute_distance(xs, ys, polygon[i], polygon[j])
        distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
    distance_map = distance_map.min(axis=0)
    
    xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
    xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
    ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
    ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
    canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
	1 - distance_map[
	    ymin_valid-ymin:ymax_valid-ymax+height,
	    xmin_valid-xmin:xmax_valid-xmax+width],
	canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])


def compute_distance(xs, ys, point_1, point_2):
    '''
    compute the distance from point to a line
    ys: coordinates in the first axis
    xs: coordinates in the second axis
    point_1, point_2: (x, y), the end of the line
    '''
    height, width = xs.shape[:2]
    square_distance_1 = np.square(
	xs - point_1[0]) + np.square(ys - point_1[1])
    square_distance_2 = np.square(
	xs - point_2[0]) + np.square(ys - point_2[1])
    square_distance = np.square(
	point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

    cosin = (square_distance - square_distance_1 - square_distance_2) / \
	(2 * np.sqrt(square_distance_1 * square_distance_2))
    square_sin = 1 - np.square(cosin)
    square_sin = np.nan_to_num(square_sin)
    result = np.sqrt(square_distance_1 * square_distance_2 *
		     square_sin / square_distance)

    result[cosin < 0] = np.sqrt(np.fmin(
	square_distance_1, square_distance_2))[cosin < 0]
    # self.extend_line(point_1, point_2, result)
    return result


class BorderMasks(PolygonMasks):
    """
    This class stores borders of for all objects in one image, forming border graduating maps.

    Attributes:
        polygons: list[list[ndarray]]. Each ndarray is a float64 vector representing a polygon.
    """

    def border_masks(
        self,
        mask:np.ndarray=None,
        mask_size:Union[int,tuple,None]=None
    ) -> np.ndarray:
        """
        Generate masks inside polygons with gradient value.
        """

        assert not (mask and mask_size), "Only one of mask and size should be specified."
        if mask is None:
            mask_size = make_dual(mask_size)
            mask = np.zeros(mask_size, dtype=np.float32)
        return self.mask_inside_polygons(mask)

    def mask_inside_polygons(
        self,
        mask:np.ndarray,
        expansion_ratio:float=0.1
    ) -> np.ndarray:
        bboxes = self.get_bounding_boxes()
        for polygons_per_instance, bbox in zip(self.polygons, bboxes):
            polygon_points = np.concatenate(polygons_per_instance, axis=0).reshape(-1, 2)
            # 1. Convet polygon to convex hull.
            hull = cv2.convexHull(polygon_points.astype(np.float32), clockwise=False)
            # (N, 1, 2)
            if hull.shape[0] < 3:
                continue

            hull = hull.reshape(-1, 2)

            dilated_hull = dilate_polygon(hull, np.sqrt(polygon_area(hull[:, 0], hull[:, 1])) * expansion_ratio)
            instance_width = int(dilated_hull[:, 0].max() - dilated_hull[:, 0].min() + 1 - 1e-5)
            instance_height = int(dilated_hull[:, 1].max() - dilated_hull[:, 1].min() + 1 - 1e-5)
            mask_for_instance = np.zeros((instance_height, instance_width), dtype=np.float32)

            # Perform rendering on cropped areas to save computation cost.
            shift = dilated_hull.min(0)
            polygon_points = polygon_points - shift
            hull = hull - shift
            dilated_hull = dilated_hull - shift
            point_o = hull.mean(axis=0)

            # 2. Draw l2 distance_map 
            draw_border_map(hull, mask_for_instance, expansion_ratio)
            # hull = dilate_polygon(hull, np.sqrt(polygon_area(hull[:, 0], hull[:, 1])) * expansion_ratio * 0.3)
            # cv2.fillPoly(mask_for_instance, [(hull-1).astype(np.int32)], 0)

            # 3. Draw l1 distance in areas for each neighboring point pairs
            point_x = hull[0]
            for next_i in range(1, hull.shape[0]):
                point_y = hull[next_i]
                local = coordinate_transform(
                    standard_linear(),
                    point_o, point_x, point_y,
                    (instance_width, instance_height))
                mask_for_instance = np.maximum(mask_for_instance, local)
                point_x = point_y

            point_y = hull[0]
            local = coordinate_transform(
                standard_linear(),
                point_o, point_x, point_y,
                (instance_width, instance_height))
            mask_for_instance = np.maximum(mask_for_instance, np.clip(local, 0, 1))

            # 4. Attach to the mask for whole image
            xmin, ymin = shift
            xmax = xmin + instance_width
            ymax = ymin + instance_height
            xmin_valid = min(max(0, xmin), mask.shape[1] - 1)
            xmax_valid = min(max(0, xmax), mask.shape[1] - 1)
            ymin_valid = min(max(0, ymin), mask.shape[0] - 1)
            ymax_valid = min(max(0, ymax), mask.shape[0] - 1)
            mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid] = np.fmax(
                mask_for_instance[
                    ymin_valid-ymin:ymax_valid-ymax+instance_height,
                    xmin_valid-xmin:xmax_valid-xmax+instance_width],
                mask[ymin_valid:ymax_valid, xmin_valid:xmax_valid])

        return mask
