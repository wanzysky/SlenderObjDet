import math
import numpy as np
from typing import List
import torch
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.layers import ShapeSpec, batched_nms, cat, DeformConv
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage

from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.retinanet import RetinaNet, permute_to_N_HWA_K

from slender_det.modeling.meta_arch.reppoints import flat_and_concate_levels
from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid

INF = 100000000

__all__ = ["CenternessRetinaNet"]


@META_ARCH_REGISTRY.register()
class CenternessRetinaNet(RetinaNet):

    # using nearest assign
    @torch.no_grad()
    def label_anchors(self, anchors, gt_instances):
        """
        Args:
            anchors (list[Boxes]): A list of #feature level Boxes.
                The Boxes contains anchors of this image on the specific feature level.
            gt_instances (list[Instances]): a list of N `Instances`s. The i-th
                `Instances` contains the ground-truth per-instance annotations
                for the i-th input image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps (sum(Hi * Wi * A)).
                Label values are in {-1, 0, ..., K}, with -1 means ignore, and K means background.
            list[Tensor]:
                i-th element is a Rx4 tensor, where R is the total number of anchors across
                feature maps. The values are the matched gt boxes for each anchor.
                Values are undefined for those anchors not labeled as foreground.
        """

        # generate strides: [R]
        points = []
        for i in range(len(anchors)):
            anchor = anchors[i].tensor
            center = torch.stack(((anchor[:, 0] + anchor[:, 2]) // 2, (anchor[:, 1] + anchor[:, 3]) // 2), dim=1)
            points.append(center)

        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, INF],
        ]
        expanded_object_sizes_of_interest = []
        for l, points_per_level in enumerate(points):
            object_sizes_of_interest_per_level = \
                points_per_level.new_tensor(object_sizes_of_interest[l])
            expanded_object_sizes_of_interest.append(
                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
            )
        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)

        gt_classes, reg_targets = compute_targets_for_locations(
            points, gt_instances, expanded_object_sizes_of_interest,
            [8, 16, 32, 64, 128], 0, 80, False
        )
        return gt_classes, reg_targets


def compute_targets_for_locations(
        locations, targets, object_sizes_of_interest,
        strides, center_sampling_radius, num_classes, norm_reg_targets=False
):
    num_points = [len(_) for _ in locations]
    # build normalization weights before cat locations
    norm_weights = None
    if norm_reg_targets:
        norm_weights = torch.cat([torch.empty(n).fill_(s) for n, s in zip(num_points, strides)])

    locations = torch.cat(locations, dim=0)
    xs, ys = locations[:, 0], locations[:, 1]

    gt_classes = []
    reg_targets = []
    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        bboxes = targets_per_im.gt_boxes.tensor
        gt_classes_per_im = targets_per_im.gt_classes
        area = targets_per_im.gt_boxes.area()

        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        # [len(locations),len(bboxes),4]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

        max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

        # [len(locations),len(bboxes)]
        locations_to_gt_area = area[None].repeat(len(locations), 1)
        locations_to_gt_area[is_in_boxes == 0] = INF
        locations_to_gt_area[is_cared_in_the_level == 0] = INF

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

        gt_classes_per_im = gt_classes_per_im[locations_to_gt_inds]
        # NOTE: set background labels to NUM_CLASSES not 0
        gt_classes_per_im[locations_to_min_area == INF] = num_classes

        # calculate regression targets in 'fcos' type
        bboxes_per_im = bboxes[None].repeat(len(locations), 1, 1)
        reg_targets_per_im = bboxes_per_im[range(len(locations)), locations_to_gt_inds]
        # reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
        if norm_reg_targets and norm_weights is not None:
            reg_targets_per_im /= norm_weights[:, None]

        gt_classes.append(gt_classes_per_im)
        reg_targets.append(reg_targets_per_im)

    return gt_classes, reg_targets
