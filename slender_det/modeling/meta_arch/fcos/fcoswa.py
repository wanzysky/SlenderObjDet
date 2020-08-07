import os
import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec, batched_nms, DeformConv
from slender_det.modeling.detector_postprocessing_with_anchor import detector_postprocess_with_anchor
from slender_det.modeling.meta_arch import FCOS

@META_ARCH_REGISTRY.register()
class FCOSWithAnchor(FCOS):

    def inference_single_feature_map(self, locations, box_cls, box_regression, centerness, image_sizes):
        N, C, H, W = box_cls.shape

        # put in the same format as locations
        box_cls = box_cls.view(N, C, H, W).permute(0, 2, 3, 1)
        box_cls = box_cls.reshape(N, -1, C).sigmoid()
        box_regression = box_regression.view(N, 4, H, W).permute(0, 2, 3, 1)
        box_regression = box_regression.reshape(N, -1, 4)
        centerness = centerness.view(N, 1, H, W).permute(0, 2, 3, 1)
        centerness = centerness.reshape(N, -1).sigmoid()

        candidate_inds = box_cls > self.pre_nms_thresh
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1)
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        # multiply the classification scores with centerness scores
        box_cls = box_cls * centerness[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = box_cls[i]
            per_candidate_inds = candidate_inds[i]
            per_box_cls = per_box_cls[per_candidate_inds]

            per_candidate_nonzeros = per_candidate_inds.nonzero()
            per_box_loc = per_candidate_nonzeros[:, 0]
            per_class = per_candidate_nonzeros[:, 1]

            per_box_regression = box_regression[i]
            per_box_regression = per_box_regression[per_box_loc]
            per_locations = locations[per_box_loc]

            per_pre_nms_top_n = pre_nms_top_n[i]

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_box_regression[:, 0],
                per_locations[:, 1] - per_box_regression[:, 1],
                per_locations[:, 0] + per_box_regression[:, 2],
                per_locations[:, 1] + per_box_regression[:, 3],
            ], dim=1)
            
            anchors = torch.stack([
                per_locations[:, 0],
                per_locations[:, 1],
                per_locations[:, 0],
                per_locations[:, 1],
            ], dim=1)
            result = Instances(image_sizes[i])
            result.pred_boxes = Boxes(detections)
            result.anchors = Boxes(anchors)
            result.scores = torch.sqrt(per_box_cls)
            result.pred_classes = per_class
            results.append(result)

        return results

    def postprocess(self, instances, batched_inputs, image_sizes):
        """
            Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            boxes = results_per_image.pred_boxes.tensor
            scores = results_per_image.scores
            class_idxs = results_per_image.pred_classes

            # Apply per-class nms for each image
            keep = batched_nms(boxes, scores, class_idxs, self.nms_thresh)
            keep = keep[: self.max_detections_per_image]
            results_per_image = results_per_image[keep]

            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess_with_anchor(results_per_image, height, width)
            processed_results.append({"instances": r})

        return processed_results
