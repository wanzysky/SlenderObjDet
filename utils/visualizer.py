import math

import numpy as np
import matplotlib as mpl
import cv2

from detectron2.utils.visualizer import Visualizer as D2Visualizer
from detectron2.utils.visualizer import VisImage
from detectron2.structures import Boxes, BoxMode, Instances

from concern.support import between, all_the_same


class Visualizer(D2Visualizer):

    def smart_concatenate(self,
                          images,
                          min_side: int = None,
                          out_shape=None) -> np.ndarray:
        assert all_the_same([img.shape for img in images])

        num_items = len(images)
        canvas = np.zeros_like(images[0])
        if min_side is not None:
            assert out_shape is None, "out_shape has been specified"
            ratio = canvas.shape[0] / canvas.shape[1]
            if ratio < 1:
                out_shape = (min_side, int(min_side / ratio + 0.5))
            else:
                out_shape = (int(min_side * ratio + 0.5), min_side)

        if out_shape is not None:
            canvas = cv2.resize(canvas, out_shape[::-1])

        num_colums = int(math.sqrt(num_items))
        num_rows = int(math.ceil(num_items / num_colums))
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

    def draw_box(self, box_coord, alpha=0.5, edge_color="g", line_style="-"):
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

        linewidth = 1

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
