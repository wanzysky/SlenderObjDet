import math
from collections import defaultdict

import numpy as np
import matplotlib as mpl
import cv2
import torch

from detectron2.utils.visualizer import Visualizer as D2Visualizer
from detectron2.utils.visualizer import ColorMode, _create_text_labels, _SMALL_OBJECT_AREA_THRESH
from detectron2.utils.visualizer import VisImage
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.colormap import random_color
from detectron2.structures.boxes import pairwise_iou

from concern.support import between, all_the_same, any_of


class Visualizer(D2Visualizer):
    def __init__(self, img_rgb, metadata, scale=1.0, instance_mode=ColorMode.IMAGE):
        self.img = np.asarray(img_rgb).clip(0, 255).astype(np.uint8)
        self.metadata = metadata
        self.output = VisImage(self.img, scale=scale)
        self.cpu_device = torch.device("cpu")

        self._default_font_size = max(
            np.sqrt(self.output.height * self.output.width) // 90, 10 // scale
        )
        self._instance_mode = instance_mode

    def smart_concatenate(self,
                          images,
                          min_side: int = None,
                          num_rows: int = None,
                          num_colums: int = None,
                          out_shape=None) -> np.ndarray:
        assert all_the_same([img.shape for img in images])

        num_items = len(images)
        if num_rows is None and num_colums is None:
            num_colums = int(math.sqrt(num_items))
            num_rows = int(math.ceil(num_items / num_colums))
        elif num_rows is None:
            num_rows = int(math.ceil(num_items / num_colums))
        elif num_colums is None:
            num_colums = int(math.ceil(num_items / num_rows))

        canvas = np.zeros_like(images[0])
        if min_side is not None:
            assert out_shape is None, "out_shape has been specified"
            ratio = (canvas.shape[0] * num_rows) / (canvas.shape[1] * num_colums)
            if ratio < 1:
                out_shape = (min_side, int(min_side / ratio + 0.5))
            else:
                out_shape = (int(min_side * ratio + 0.5), min_side)

        if out_shape is not None:
            canvas = cv2.resize(canvas, out_shape[::-1])

        h, w = int(canvas.shape[0] / num_rows), int(canvas.shape[1] / num_colums)

        for r in range(num_rows):
            for c in range(num_colums):
                if not r * num_colums + c < len(images):
                    break
                canvas[r * h:(r + 1) * h, c * w:(c + 1)
                       * w] = cv2.resize(images[r * num_colums + c], (w, h))
        return canvas

    def draw_proposals_separately(self, proposals, image_shape, conf_threshold):
        ratios_ranges = [(0, 0.6), (0.9, 1.2), (1.5, 2.2)]
        width_ranges = [32, 64, 128, 256, 512]
        previous_mark = 0
        area_ranges = []
        for w in width_ranges:
            area_ranges.append((previous_mark ** 2 + 4, w ** 2 + 64))
            previous_mark = w

        boxes = np.asarray([x["bbox"] for x in proposals]).reshape(-1, 4)
        box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        box_ratios = (boxes[:, 0] - boxes[:, 2]) / (boxes[:, 1] - boxes[:, 3])
        scores = np.asarray([x["score"] for x in proposals])

        output_images = []
        for r_r in ratios_ranges:
            # a_r = area_ranges[0]
            # if True:
            for a_r in area_ranges:
                instance = Instances(image_shape)
                chosen = np.logical_and(
                    between(box_ratios, r_r),
                    between(box_area, a_r))
                chosen = np.logical_and(chosen, scores > conf_threshold).nonzero()[0]
                score = scores[chosen]
                bbox = np.asarray([proposals[i]["bbox"] for i in chosen]).reshape(-1, 4)
                bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)

                instance.scores = score
                instance.pred_boxes = Boxes(bbox)
                output_images.append(self.draw_proposals(instance))
        return output_images

    def draw_proposals(self, proposals):
        img = self.img.copy()
        labels = [str(round(score, 2)) for score in proposals.scores]
        self.overlay_instances(boxes=proposals.pred_boxes, labels=labels, alpha=1)
        rendered = self.output.get_image()
        self.img = img
        self.output = VisImage(self.img)
        return rendered

    def topk_iou_boxes(self, candidates: Boxes, targets: Boxes, k=1):
        iou_matrix = pairwise_iou(candidates, targets)
        _, topk_idxs = iou_matrix.topk(k, dim=0)
        return candidates.tensor[topk_idxs], topk_idxs

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-", linewidth=2):
        """
        Args:
            box_coord (tuple): a tuple containing x0, y0, x1, y1 coordinates, where x0 and y0
                are the coordinates of the image's top left corner. x1 and y1 are the
                coordinates of the image's bottom right corner.
            alpha (float): blending efficient. Smaller values lead to more transparent masks.
            edge_color: color of the outline of the box. Refer to `matplotlib.colors`
                for full list of formats that are accepted.
            line_style (string): the string to use to create the outline of the boxes.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x0, y0, x1, y1 = box_coord
        width = x1 - x0
        height = y1 - y0

        self.output.ax.add_patch(
            mpl.patches.Rectangle(
                (x0, y0),
                width,
                height,
                fill=False,
                edgecolor=edge_color,
                linewidth=linewidth * self.output.scale,
                alpha=alpha,
                linestyle=line_style,
            )
        )
        return self.output

    def group_by(self, object_list, values, ranges_dic):
        """
        Group given boxes by judging whether values are bwtween
        the ranges in ranges_dic. A box can be assigned to multiple 
        groups as there are no regulations on the intersections of ranges.
        """
        grouped = defaultdict(list)
        assert len(object_list) == len(values)

        for index, item in enumerate(object_list):
            for key, ranges in ranges_dic.items():
                # An item in ranges dic may be a list of conditions.
                if any_of(ranges, between, values[index]):
                    grouped[key].append(item)
        return grouped

    def select_dataset_dict_by_ratio(self, anno, ratios_ranges):

        boxes = np.asarray([x["bbox"] for x in predictions]).reshape(-1, 4)
        boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        box_ratios = (boxes[:, 0] - boxes[:, 2]) / (boxes[:, 1] - boxes[:, 3])
        scores = np.asarray([x["score"] for x in predictions])

        selected_predictions = []
        for r_r in ratios_ranges:
            chosen = between(box_ratios, r_r).nonzero()[0]

            chosen_predictions = np.asarray(predictions)[chosen]
            selected_predictions.append(chosen_predictions)
        return selected_predictions

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_circle(self, circle_coord, color, radius=6):
        """
        Args:
            circle_coord (list(int) or tuple(int)): contains the x and y coordinates
                of the center of the circle.
            color: color of the polygon. Refer to `matplotlib.colors` for a full list of
                formats that are accepted.
            radius (int): radius of the circle.

        Returns:
            output (VisImage): image object with box drawn.
        """
        x, y = circle_coord
        self.output.ax.add_patch(
            mpl.patches.Circle(circle_coord, radius=radius, fill=True, color=color)
        )
        return self.output

    def overlay_instances(
        self,
        *,
        boxes=None,
        labels=None,
        masks=None,
        keypoints=None,
        assigned_colors=None,
        alpha=0.5
    ):
        """
        Args:
            boxes (Boxes, RotatedBoxes or ndarray): either a :class:`Boxes`,
                or an Nx4 numpy array of XYXY_ABS format for the N objects in a single image,
                or a :class:`RotatedBoxes`,
                or an Nx5 numpy array of (x_center, y_center, width, height, angle_degrees) format
                for the N objects in a single image,
            labels (list[str]): the text to be displayed for each instance.
            masks (masks-like object): Supported types are:

                * :class:`detectron2.structures.PolygonMasks`,
                  :class:`detectron2.structures.BitMasks`.
                * list[list[ndarray]]: contains the segmentation masks for all objects in one image.
                  The first level of the list corresponds to individual instances. The second
                  level to all the polygon that compose the instance, and the third level
                  to the polygon coordinates. The third level should have the format of
                  [x0, y0, x1, y1, ..., xn, yn] (n >= 3).
                * list[ndarray]: each ndarray is a binary mask of shape (H, W).
                * list[dict]: each dict is a COCO-style RLE.
            keypoints (Keypoint or array like): an array-like object of shape (N, K, 3),
                where the N is the number of instances and K is the number of keypoints.
                The last dimension corresponds to (x, y, visibility or score).
            assigned_colors (list[matplotlib.colors]): a list of colors, where each color
                corresponds to each mask or box in the image. Refer to 'matplotlib.colors'
                for full list of formats that the colors are accepted in.

        Returns:
            output (VisImage): image object with visualizations.
        """
        num_instances = None
        if boxes is not None:
            boxes = self._convert_boxes(boxes)
            num_instances = len(boxes)
        if masks is not None:
            masks = self._convert_masks(masks)
            if num_instances:
                assert len(masks) == num_instances
            else:
                num_instances = len(masks)
        if keypoints is not None:
            if num_instances:
                assert len(keypoints) == num_instances
            else:
                num_instances = len(keypoints)
            keypoints = self._convert_keypoints(keypoints)
        if labels is not None:
            assert len(labels) == num_instances
        if assigned_colors is None:
            assigned_colors = [random_color(rgb=True, maximum=1) for _ in range(num_instances)]
        if num_instances == 0:
            return self.output
        if boxes is not None and boxes.shape[1] == 5:
            return self.overlay_rotated_instances(
                boxes=boxes, labels=labels, assigned_colors=assigned_colors
            )

        # Display in largest to smallest order to reduce occlusion.
        areas = None
        if boxes is not None:
            areas = np.prod(boxes[:, 2:] - boxes[:, :2], axis=1)
        elif masks is not None:
            areas = np.asarray([x.area() for x in masks])

        if areas is not None:
            sorted_idxs = np.argsort(-areas).tolist()
            # Re-order overlapped instances in descending order.
            boxes = boxes[sorted_idxs] if boxes is not None else None
            labels = [labels[k] for k in sorted_idxs] if labels is not None else None
            masks = [masks[idx] for idx in sorted_idxs] if masks is not None else None
            assigned_colors = [assigned_colors[idx] for idx in sorted_idxs]
            keypoints = keypoints[sorted_idxs] if keypoints is not None else None

        for i in range(num_instances):
            color = assigned_colors[i]
            if boxes is not None:
                if boxes[i, 0] - boxes[i, 2] == 0 and boxes[i, 1] - boxes[i, 3] == 0:
                    self.draw_circle(boxes[i, :2], color)
                else:
                    self.draw_box(boxes[i], edge_color=color)

            if masks is not None:
                for segment in masks[i].polygons:
                    self.draw_polygon(segment.reshape(-1, 2), color, alpha=alpha)

            if labels is not None:
                # first get a box
                if boxes is not None:
                    x0, y0, x1, y1 = boxes[i]
                    text_pos = (x0, y0)  # if drawing boxes, put text on the box corner.
                    horiz_align = "left"
                elif masks is not None:
                    x0, y0, x1, y1 = masks[i].bbox()

                    # draw text in the center (defined by median) when box is not drawn
                    # median is less sensitive to outliers.
                    text_pos = np.median(masks[i].mask.nonzero(), axis=1)[::-1]
                    horiz_align = "center"
                else:
                    continue  # drawing the box confidence for keypoints isn't very useful.
                # for small objects, draw text at the side to avoid occlusion
                instance_area = (y1 - y0) * (x1 - x0)
                if (
                    instance_area < _SMALL_OBJECT_AREA_THRESH * self.output.scale
                    or y1 - y0 < 40 * self.output.scale
                ):
                    if y1 >= self.output.height - 5:
                        text_pos = (x1, y0)
                    else:
                        text_pos = (x0, y1)

                height_ratio = (y1 - y0) / np.sqrt(self.output.height * self.output.width)
                lighter_color = self._change_color_brightness(color, brightness_factor=0.7)
                font_size = (
                    np.clip((height_ratio - 0.02) / 0.08 + 1, 1.2, 2)
                    * 0.5
                    * self._default_font_size
                )
                self.draw_text(
                    labels[i],
                    text_pos,
                    color=lighter_color,
                    horizontal_alignment=horiz_align,
                    font_size=font_size,
                )

        # draw keypoints
        if keypoints is not None:
            for keypoints_per_instance in keypoints:
                self.draw_and_connect_keypoints(keypoints_per_instance)

        return self.output
