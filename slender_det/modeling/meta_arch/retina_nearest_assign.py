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

from slender_det.modeling.meta_arch.rpd import flat_and_concate_levels
from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid
from slender_det.modeling.matcher import rep_points_match
__all__ = ["NearestRetinaNet"]

@META_ARCH_REGISTRY.register()
class NearestRetinaNet(RetinaNet):

    #using nearest assign
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
        
        #generate strides: [R]
        strides=[]
        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        for i in range(len(feature_shapes)):
            stride = feature_shapes[i].stride
            anchor_num_i = anchors[i].tensor.shape[0]
            stride = torch.full((anchor_num_i, ), stride, device=anchors[i].tensor.device)
            strides.append(stride)
        anchors = Boxes.cat(anchors).tensor
        centers = torch.stack(((anchors[:,0]+ anchors[:,2])//2,(anchors[:,1]+ anchors[:,3])//2), dim = 1)
        strides = torch.cat(strides,0)
        
        gt_labels = []
        matched_gt_boxes = []
        for gt_per_image in gt_instances:
            image_size = gt_per_image.image_size
            centers_invalid = (centers[:, 0] >= image_size[1]).logical_or(
                centers[:, 1] >= image_size[0])
                
            objectness_label_i, bbox_label_i = rep_points_match(
                centers, strides, gt_per_image.gt_boxes, gt_per_image.gt_classes)
            objectness_label_i[objectness_label_i == -1] = self.num_classes
            objectness_label_i[centers_invalid] = -1
            gt_labels.append(objectness_label_i)
            matched_gt_boxes.append(bbox_label_i)
        return gt_labels, matched_gt_boxes
#            match_quality_matrix = pairwise_iou(gt_per_image.gt_boxes, anchors)
#            matched_idxs, anchor_labels = self.anchor_matcher(match_quality_matrix)
#            del match_quality_matrix
#
#            if len(gt_per_image) > 0:
#                matched_gt_boxes_i = gt_per_image.gt_boxes.tensor[matched_idxs]
#
#                gt_labels_i = gt_per_image.gt_classes[matched_idxs]
#                # Anchors with label 0 are treated as background.
#                gt_labels_i[anchor_labels == 0] = self.num_classes
#                # Anchors with label -1 are ignored.
#                gt_labels_i[anchor_labels == -1] = -1
#            else:
#                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
#                gt_labels_i = torch.zeros_like(matched_idxs) + self.num_classes
#
#            gt_labels.append(gt_labels_i)
#            matched_gt_boxes.append(matched_gt_boxes_i)
#
#        return gt_labels, matched_gt_boxes