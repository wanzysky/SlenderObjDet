import os
from typing import List
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes, pairwise_iou
from detectron2.modeling.matcher import Matcher
from detectron2.layers import ShapeSpec, batched_nms, cat, DeformConv
from detectron2.modeling.postprocessing import detector_postprocess

from slender_det.modeling.backbone import build_backbone
from slender_det.layers import Scale, iou_loss, DFConv2d, smooth_l1_loss_with_weight

from .utils import INF, get_num_gpus, reduce_sum, permute_to_N_HW_K, \
    compute_locations_per_level, compute_locations, get_sample_region, \
    compute_centerness_targets


def permute_and_concat(box_cls, box_reg_init, box_reg, center_score, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_reg and the center-ness
    box_cls_flattened = [permute_to_N_HW_K(x, num_classes) for x in box_cls]
    box_reg_flattened = [permute_to_N_HW_K(x, 4) for x in box_reg]
    box_reg_init_flattened = [permute_to_N_HW_K(x, 18) for x in box_reg_init]
    center_score = [permute_to_N_HW_K(x, 1) for x in center_score]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_reg = cat(box_reg_flattened, dim=1).view(-1, 4)
    box_reg_init = cat(box_reg_init_flattened, dim=1).view(-1, 18)
    center_score = cat(center_score, dim=1).view(-1)

    return box_cls, box_reg_init, box_reg, center_score


def compute_targets_for_locations(
        locations, targets, object_sizes_of_interest, strides, center_sampling_radius, num_classes
):
    num_points = [len(_) for _ in locations]

    locations = torch.cat(locations, dim=0)
    xs, ys = locations[:, 0], locations[:, 1]

    gt_classes = []
    reg_targets = []
    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        bboxes = targets_per_im.gt_boxes.tensor
        gt_classes_per_im = targets_per_im.gt_classes
        area = targets_per_im.gt_boxes.area()
        
        #transfer bboxes from 4 channel to 18 channel
        lt = bboxes[:,[0,1]]
        rt = bboxes[:,[2,1]]
        lb = bboxes[:,[0,3]]
        rb = bboxes[:,[2,3]]
        lc = torch.stack((bboxes[:,0],(bboxes[:,1]+bboxes[:,3])/2),dim=1)
        tc = torch.stack(((bboxes[:,0]+bboxes[:,2])/2,bboxes[:,1]),dim=1)
        rc = torch.stack((bboxes[:,2],(bboxes[:,1]+bboxes[:,3])/2),dim=1)
        bc = torch.stack(((bboxes[:,0]+bboxes[:,2])/2,bboxes[:,3]),dim=1)
        cc = torch.stack(((bboxes[:,0]+bboxes[:,2])/2,(bboxes[:,1]+bboxes[:,3])/2),dim=1)
        expand_bboxes = torch.cat((lt,tc,rt,lc,cc,rc,lb,bc,rb),dim=1)
        xys = locations.repeat(1,9)
        offsets_per_im = xys[:,None] - expand_bboxes
        
        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        if center_sampling_radius > 0:
            is_in_boxes = get_sample_region(bboxes, strides, num_points, xs, ys, radius=center_sampling_radius)
        else:
            # no center sampling, it will use all the locations within a ground-truth box
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

        max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

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
        offsets_per_im = offsets_per_im[range(len(locations)), locations_to_gt_inds]
        #reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

        gt_classes.append(gt_classes_per_im)
        #reg_targets.append(reg_targets_per_im)
        reg_targets.append(offsets_per_im)

    return torch.stack(gt_classes), torch.stack(reg_targets)


@META_ARCH_REGISTRY.register()
class FCOSRepPoints(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.in_features = cfg.MODEL.FCOS.IN_FEATURES

        # Loss parameters:
        # defined by method<get_ground_truth>
        self.num_points_per_level = None
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.iou_loss_type = cfg.MODEL.FCOS.IOU_LOSS_TYPE

        # Inference parameters:
        self.score_thresh = 0.3
        self.pre_nms_thresh = cfg.MODEL.FCOS.INFERENCE_TH
        self.pre_nms_top_n = cfg.MODEL.FCOS.PRE_NMS_TOP_N
        self.nms_thresh = cfg.MODEL.FCOS.NMS_TH
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
        self.min_size = 0
        self.num_classes = cfg.MODEL.FCOS.NUM_CLASSES

        self.backbone = build_backbone(cfg)

        backbone_shape = self.backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.head = FCOSRepPointsHead(cfg, feature_shapes)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.bbox_matcher = Matcher(
            cfg.MODEL.RETINANET.IOU_THRESHOLDS,
            cfg.MODEL.RETINANET.IOU_LABELS,
            allow_low_quality_matches=True,
        )

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            Same as in :class:`GeneralizedRCNN.forward`

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "proposals" whose value is a
                :class:`Instances` with keys "proposal_boxes" and "objectness_logits".
        """
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        elif "targets" in batched_inputs[0]:
            log_first_n(
                logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10
            )
            gt_instances = [x["targets"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        box_cls, box_init, box_refine, ctr_sco = self.head(features)

        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)

        # compute strides: List[[X]]
        strides = [torch.full((shapes[i][0] * shapes[i][1],), self.fpn_strides[i], device=self.device) for i in
                   range(len(shapes))]
        strides = torch.cat(strides, dim=0)
        
        offset_reg = [permute_to_N_HW_K(x, 18) for x in box_init]
        # for each level
        predicted_boxes_init = []
        for i in range(len(locations)):
            locs_i = locations[i]  # [X,2]
            offset_reg_i = offset_reg[i]  # [N,H*W,18]
            box_reg_i_x1 = offset_reg_i[:,:,::2].min(dim=2, keepdim=True)[0]
            box_reg_i_y1 = offset_reg_i[:,:,1::2].min(dim=2, keepdim=True)[0]
            box_reg_i_x2 = offset_reg_i[:,:,::2].max(dim=2, keepdim=True)[0]
            box_reg_i_y2 = offset_reg_i[:,:,1::2].max(dim=2, keepdim=True)[0]
            box_reg_i = torch.cat((box_reg_i_x1,box_reg_i_y1,box_reg_i_x2,box_reg_i_y2), dim=2)
            locs_i = locs_i.repeat(box_reg_i.shape[0], 1, 1)  # [N,X,2]
            predicted_boxes = torch.stack([
                locs_i[:, :, 0] - box_reg_i[:, :, 0], locs_i[:, :, 1] - box_reg_i[:, :, 1],
                locs_i[:, :, 0] + box_reg_i[:, :, 2], locs_i[:, :, 1] + box_reg_i[:, :, 3],
            ], dim=2)  # [N,X,4]
            predicted_boxes_init.append(predicted_boxes)
        predicted_boxes_init = torch.cat(predicted_boxes_init, dim=1)
        if self.training:
            init_gt_classes, init_reg_targets, refine_gt_classes, refine_reg_targets = \
                self.get_ground_truth(locations, predicted_boxes_init, gt_instances)
            #            gt_classes, reg_targets = self.get_ground_truth(locations, gt_instances)
            losses = self.losses(init_gt_classes, init_reg_targets, refine_gt_classes, refine_reg_targets, box_cls,
                                 box_init, box_refine, ctr_sco, strides)

            return losses
        else:
            results = self.inference(locations, box_cls, box_refine, ctr_sco, images.image_sizes)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def losses(self, init_gt_classes, init_reg_targets, refine_gt_classes, refine_reg_targets, \
               pred_class_logits, pred_box_reg_init, pred_box_reg, pred_center_score, strides):

        strides = strides.repeat(pred_class_logits[0].shape[0])  # [N*X]
        pred_class_logits, pred_box_reg_init, pred_box_reg, pred_center_score = \
            permute_and_concat(pred_class_logits, pred_box_reg_init, pred_box_reg, pred_center_score, self.num_classes)
        # Shapes: (N x R) and (N x R, 4), (N x R) respectively.

        init_gt_classes = init_gt_classes.flatten()
        init_reg_targets = init_reg_targets.view(-1, 18)

        init_foreground_idxs = (init_gt_classes >= 0) & (init_gt_classes != self.num_classes)
        init_pos_inds = torch.nonzero(init_foreground_idxs).squeeze(1)

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        init_total_num_pos = reduce_sum(init_pos_inds.new_tensor([init_pos_inds.numel()])).item()
        init_num_pos_avg_per_gpu = max(init_total_num_pos / float(num_gpus), 1.0)

        refine_gt_classes = refine_gt_classes.flatten()
        refine_reg_targets = refine_reg_targets.view(-1, 4)

        refine_foreground_idxs = (refine_gt_classes >= 0) & (refine_gt_classes != self.num_classes)
        refine_pos_inds = torch.nonzero(refine_foreground_idxs).squeeze(1)

        # sync num_pos from all gpus
        refine_total_num_pos = reduce_sum(refine_pos_inds.new_tensor([refine_pos_inds.numel()])).item()
        refine_num_pos_avg_per_gpu = max(refine_total_num_pos / float(num_gpus), 1.0)

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[refine_foreground_idxs, refine_gt_classes[refine_foreground_idxs]] = 1

        # logits loss
        cls_loss = sigmoid_focal_loss_jit(
            pred_class_logits, gt_classes_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum",
        ) / refine_num_pos_avg_per_gpu
        
        offset_reg = init_reg_targets[init_foreground_idxs]  # [N,H*W,18]
        box_reg_x1 = offset_reg[:,::2].min(dim=1, keepdim=True)[0] * -1
        box_reg_y1 = offset_reg[:,1::2].min(dim=1, keepdim=True)[0] * -1
        box_reg_x2 = offset_reg[:,::2].max(dim=1, keepdim=True)[0]
        box_reg_y2 = offset_reg[:,1::2].max(dim=1, keepdim=True)[0]
        box_reg = torch.cat((box_reg_x1,box_reg_y1,box_reg_x2,box_reg_y2), dim=1)
        gt_center_score = compute_centerness_targets(box_reg)
        # average sum_centerness_targets from all gpus,
        # which is used to normalize centerness-weighed reg loss
        sum_centerness_targets_avg_per_gpu = \
            reduce_sum(gt_center_score.sum()).item() / float(num_gpus)
#        reg_loss_init = iou_loss(
#            pred_box_reg_init[init_foreground_idxs], init_reg_targets[init_foreground_idxs], gt_center_score,
#            loss_type=self.iou_loss_type
#        ) / sum_centerness_targets_avg_per_gpu
        coords_norm_init = strides[init_foreground_idxs].unsqueeze(-1) * 18
        reg_loss_init = smooth_l1_loss_with_weight(
            pred_box_reg_init[init_foreground_idxs] / coords_norm_init,
            init_reg_targets[init_foreground_idxs] / coords_norm_init,
            gt_center_score,
            0.11, reduction="sum") / sum_centerness_targets_avg_per_gpu * 0.5
            #0.11, reduction="sum") / max(1, init_num_pos_avg_per_gpu)

        coords_norm_refine = strides[refine_foreground_idxs].unsqueeze(-1) * 4
        reg_loss = smooth_l1_loss(
            pred_box_reg[refine_foreground_idxs] / coords_norm_refine,
            refine_reg_targets[refine_foreground_idxs] / coords_norm_refine,
            0.11, reduction="sum") / max(1, refine_num_pos_avg_per_gpu)
        #        reg_loss = iou_loss(
        #            pred_box_reg[refine_foreground_idxs], refine_reg_targets[refine_foreground_idxs], 1,
        #            loss_type=self.iou_loss_type
        #        ) / sum_centerness_targets_avg_per_gpu

        centerness_loss = F.binary_cross_entropy_with_logits(
            pred_center_score[init_foreground_idxs], gt_center_score, reduction='sum'
        ) / init_num_pos_avg_per_gpu

        return dict(cls_loss=cls_loss, reg_loss_init=reg_loss_init, reg_loss=reg_loss, centerness_loss=centerness_loss)
        # return dict(cls_loss=cls_loss, reg_loss_init=reg_loss_init, centerness_loss=centerness_loss)

    @torch.no_grad()
    def get_ground_truth(self, points: torch.Tensor, init_boxes, gt_instances):
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

        init_gt_classes, init_reg_targets = compute_targets_for_locations(
            points, gt_instances, expanded_object_sizes_of_interest,
            self.fpn_strides, self.center_sampling_radius, self.num_classes
        )

        centers = torch.cat(points, 0)  # [X,2]

        cls_labels = []
        refine_bbox_labels = []
        for i, targets_per_image in enumerate(gt_instances):
            image_size = targets_per_image.image_size
            centers_invalid = (centers[:, 0] >= image_size[1]).logical_or(
                centers[:, 1] >= image_size[0])

            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes,
                Boxes(init_boxes[i]))
            gt_matched_idxs, bbox_mached = self.bbox_matcher(match_quality_matrix)
            cls_label = targets_per_image.gt_classes[gt_matched_idxs]
            cls_label[bbox_mached == 0] = self.num_classes
            cls_label[centers_invalid] = -1
            refine_bbox_label = targets_per_image.gt_boxes[gt_matched_idxs]

            # change bbox to ltrb
            refine_bbox_label = refine_bbox_label.tensor  # [X,4]
            xs, ys = centers[:, 0], centers[:, 1]
            l = xs - refine_bbox_label[:, 0]
            t = ys - refine_bbox_label[:, 1]
            r = refine_bbox_label[:, 2] - xs
            b = refine_bbox_label[:, 3] - ys
            refine_bbox_label = torch.stack([l, t, r, b], dim=1)

            cls_labels.append(cls_label)
            refine_bbox_labels.append(refine_bbox_label)

        refine_gt_classes = torch.stack(cls_labels)
        refine_reg_targets = torch.stack(refine_bbox_labels)

        return init_gt_classes, init_reg_targets, refine_gt_classes, refine_reg_targets

    #    @torch.no_grad()
    #    def get_ground_truth(self, points, gt_instances):
    #        object_sizes_of_interest = [
    #            [-1, 64],
    #            [64, 128],
    #            [128, 256],
    #            [256, 512],
    #            [512, INF],
    #        ]
    #        expanded_object_sizes_of_interest = []
    #        for l, points_per_level in enumerate(points):
    #            object_sizes_of_interest_per_level = \
    #                points_per_level.new_tensor(object_sizes_of_interest[l])
    #            expanded_object_sizes_of_interest.append(
    #                object_sizes_of_interest_per_level[None].expand(len(points_per_level), -1)
    #            )
    #        expanded_object_sizes_of_interest = torch.cat(expanded_object_sizes_of_interest, dim=0)
    #
    #        gt_classes, reg_targets = compute_targets_for_locations(
    #            points, gt_instances, expanded_object_sizes_of_interest,
    #            self.fpn_strides, self.center_sampling_radius, self.num_classes, self.norm_reg_targets
    #        )
    #        return gt_classes, reg_targets

    def inference(self, locations, box_cls, box_reg, ctr_sco, image_sizes):
        results = []

        box_cls = [permute_to_N_HW_K(x, self.num_classes) for x in box_cls]
        box_reg = [permute_to_N_HW_K(x, 4) for x in box_reg]
        ctr_sco = [permute_to_N_HW_K(x, 1) for x in ctr_sco]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_reg]
            ctr_sco_per_image = [ctr_sco_per_level[img_idx] for ctr_sco_per_level in ctr_sco]

            results_per_image = self.inference_single_image(
                locations, box_cls_per_image, box_reg_per_image, ctr_sco_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

    def inference_single_image(self, locations, box_cls, box_reg, center_score, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, locs_i, center_score_i in zip(box_cls, box_reg, locations, center_score):
            # (HxW, C)
            box_cls_i = box_cls_i.sigmoid_()
            keep_idxs = box_cls_i > self.pre_nms_thresh

            # multiply the classification scores with center scores
            box_cls_i *= center_score_i.sigmoid_()

            box_cls_i = box_cls_i[keep_idxs]
            keep_idxs_nonzero_i = keep_idxs.nonzero()

            box_loc_i = keep_idxs_nonzero_i[:, 0]
            class_i = keep_idxs_nonzero_i[:, 1]

            box_reg_i = box_reg_i[box_loc_i]
            locs_i = locs_i[box_loc_i]

            per_pre_nms_top_n = keep_idxs.sum().clamp(max=self.pre_nms_top_n)
            if keep_idxs.sum().item() > per_pre_nms_top_n.item():
                box_cls_i, topk_idxs = box_cls_i.topk(per_pre_nms_top_n, sorted=False)

                class_i = class_i[topk_idxs]
                box_reg_i = box_reg_i[topk_idxs]
                locs_i = locs_i[topk_idxs]

            # predict boxes
            predicted_boxes = torch.stack([
                locs_i[:, 0] - box_reg_i[:, 0], locs_i[:, 1] - box_reg_i[:, 1],
                locs_i[:, 0] + box_reg_i[:, 2], locs_i[:, 1] + box_reg_i[:, 3],
            ], dim=1)
            box_cls_i = torch.sqrt(box_cls_i)

            boxes_all.append(predicted_boxes)
            scores_all.append(box_cls_i)
            class_idxs_all.append(class_i)

        boxes_all, scores_all, class_idxs_all = [
            cat(x) for x in [boxes_all, scores_all, class_idxs_all]
        ]

        # Apply per-class nms for each image
        keep = batched_nms(boxes_all, scores_all, class_idxs_all, self.nms_thresh)
        keep = keep[: self.max_detections_per_image]

        result = Instances(image_size)
        result.pred_boxes = Boxes(boxes_all[keep])
        result.scores = scores_all[keep]
        result.pred_classes = class_idxs_all[keep]

        return result

    def postprocess(self, instances, batched_inputs, image_sizes):
        """
            Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})

        return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class FCOSRepPointsHead(torch.nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSRepPointsHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        # fmt: off
        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.FCOS.NUM_CLASSES
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS
        self.centerness_on_reg = cfg.MODEL.FCOS.CENTERNESS_ON_REG
        self.use_dcn_in_tower = cfg.MODEL.FCOS.USE_DCN_IN_TOWER
        self.use_dcn_v2 = cfg.MODEL.FCOS.USE_DCN_V2
        # fmt: on

        cls_tower = []
        bbox_tower = []
        for i in range(cfg.MODEL.FCOS.NUM_CONVS):
            use_dcn = False
            use_v2 = True
            if self.use_dcn_in_tower and i == cfg.MODEL.FCOS.NUM_CONVS - 1:
                conv_func = DFConv2d
                bias = False
                use_dcn = True
                if not self.use_dcn_v2:
                    use_v2 = False
            else:
                conv_func = nn.Conv2d
                bias = True

            if use_dcn and not use_v2:
                cls_tower.append(
                    conv_func(
                        in_channels, in_channels,
                        with_modulated_dcn=False, kernel_size=3, stride=1, padding=1, bias=bias
                    )
                )
            else:
                cls_tower.append(
                    conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                )
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())

            if use_dcn and not use_v2:
                bbox_tower.append(
                    conv_func(
                        in_channels, in_channels,
                        with_modulated_dcn=False, kernel_size=3, stride=1, padding=1, bias=bias
                    )
                )
            else:
                bbox_tower.append(
                    conv_func(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=bias)
                )
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))

        # rep part
        self.point_feat_channels = in_channels
        self.num_points = 9
        self.dcn_kernel = int(np.sqrt(self.num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        self.cls_out_channels = num_classes
        self.gradient_mul = 0.1
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset, dtype=torch.float32).view(1, -1, 1, 1)
        self.register_buffer("dcn_base_offset", dcn_base_offset)

        self.deform_cls_conv = DeformConv(
            self.point_feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1, self.dcn_pad)
        self.deform_reg_conv = DeformConv(
            self.point_feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1, self.dcn_pad)

        points_out_dim = 2 * self.num_points
        self.offsets_init = nn.Sequential(
            nn.Conv2d(self.point_feat_channels,
                      self.point_feat_channels,
                      3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.point_feat_channels,
                      points_out_dim,
                      1, 1, 0))

        self.offsets_refine = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.point_feat_channels,
                      points_out_dim,
                      1, 1, 0))
        self.logits = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.point_feat_channels,
                      self.cls_out_channels,
                      1, 1, 0))
        #        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        #        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        #                        self.cls_logits, self.bbox_pred,
                        self.offsets_init,
                        self.offsets_refine,
                        self.deform_cls_conv,
                        self.deform_reg_conv,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        #        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        for module in self.logits.modules():
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.constant_(module.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        cls_features = []
        reg_features = []
        bbox_reg = []
        centerness = []
        logits = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            # logits.append(self.logits(cls_tower))
            cls_features.append(cls_tower)
            reg_features.append(box_tower)
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            # bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            bbox_pred = self.scales[l](self.offsets_init(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                # bbox_reg.append(torch.exp(bbox_pred))

                # not ltrb, but offset to center, including negative values.
                bbox_reg.append(bbox_pred)
        # rep part
        offsets_init = bbox_reg

        offsets_refine = []
        for i in range(len(offsets_init)):
            pts_out_init_grad_mul = (1 - self.gradient_mul) * offsets_init[i].detach() \
                                    + self.gradient_mul * offsets_init[i]
            # N, 18, H, W --> N, 9, 2(x, y), H, W --> N, 9, 2(y, x), H, W
            # BUGGY: assuming self.num_points == 9
            pts_out_init_grad_mul = pts_out_init_grad_mul.reshape(
                pts_out_init_grad_mul.size(0),
                9, 2,
                *pts_out_init_grad_mul.shape[-2:]
            ).flip(2)
            pts_out_init_grad_mul = pts_out_init_grad_mul.reshape(
                -1, 18, *pts_out_init_grad_mul.shape[-2:])
            dcn_offset = pts_out_init_grad_mul - self.dcn_base_offset

            logits.append(
                self.logits(self.deform_cls_conv(cls_features[i], dcn_offset)))
            offsets_refine.append(
                self.offsets_refine(
                    self.deform_reg_conv(reg_features[i], dcn_offset)) +
                offsets_init[i].detach())
        
        #use 9 points offsets_init as output
        point_strides=[1, 2, 4, 8, 16]
        for i in range(len(offsets_init)):
            offsets_init[i] = offsets_init[i] * point_strides[i]
        # reshape the tensor from 9 points to 2 points
        #offsets_init = self.offsets2ltrb(offsets_init)
        offsets_refine = self.offsets2ltrb(offsets_refine)


        return logits, offsets_init, offsets_refine, centerness

    def offsets2ltrb(self, deltas: List[torch.Tensor], point_strides=[1, 2, 4, 8, 16]):
        '''
            Args:
                base_grids: List[[H*W,2]] coordinate of each feature map
                deltas: List[[N,C,H,W]] offsets
            Returns:
                bboxes: List[[N,4,H,W]]
        '''
        bboxes = []
        # For each level
        for i in range(len(deltas)):
            """
            delta: (N, C, H_i, W_i),
            C=4 or 18 
            """
            delta = deltas[i]
            N, C, H_i, W_i = delta.shape
            # (1, 2, H_i, W_i), grid for this feature level.
            # base_grid = base_grids[i].view(1, H_i, W_i, 2).permute(0, 3, 1, 2)

            # (N*C/2, 2, H_i, W_i)
            delta = delta.view(-1, C // 2, 2, H_i, W_i)
            # (N, C/2, 2, H_i, W_i)
            points = delta * point_strides[i]
            pts_x = points[:, :, 0, :, :]
            pts_y = points[:, :, 1, :, :]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]

            # change to ltrb format
            bbox = torch.cat(
                [bbox_left * (-1), bbox_up * (-1), bbox_right, bbox_bottom],
                dim=1)
            bboxes.append(bbox)
        return bboxes
