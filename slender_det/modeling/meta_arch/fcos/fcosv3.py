import os
import math
from typing import List

import numpy as np
from concern.smart_path import smart_path
from os.path import exists
from os import mknod
import cv2
from PIL import Image, ImageDraw

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from fvcore.nn import sigmoid_focal_loss_jit

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.modeling.postprocessing import detector_postprocess

from slender_det.modeling.backbone import build_backbone
from slender_det.layers import Scale, iou_loss, DFConv2d

from .fcosv2 import FCOSHead
from .utils import INF, reduce_sum, permute_to_N_HW_K, permute_and_concat, \
    compute_locations_per_level, compute_locations, get_sample_region, \
    compute_targets_for_locations, compute_centerness_targets


def save(rgb: np.ndarray, mask: np.ndarray, centers: np.ndarray, *file_names):
    assert len(file_names) > 0
    rgb_save_path = smart_path(os.path.join(file_names[0], file_names[1]))
    mask_save_path = smart_path(os.path.join(file_names[0], file_names[2]))
    if not exists(rgb_save_path):
        mknod(rgb_save_path)
    if not exists(mask_save_path):
        mknod(mask_save_path)

    im_rgb = Image.fromarray(rgb)
    im_rgb.save(rgb_save_path)

    mask = cv2.applyColorMap(mask, 0)

    im_mask = Image.fromarray(mask)
    draw = ImageDraw.Draw(im_mask)
    radis = 2
    for i in range(len(centers)):
        draw.pieslice((centers[i, 0] - radis, centers[i, 1] - radis, centers[i, 0] + radis, centers[i, 1] + radis),
                      start=0, end=360, fill=128)
    im_mask.save(mask_save_path)


@META_ARCH_REGISTRY.register()
class FCOSV3(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
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
        self.head = FCOSHead(cfg, feature_shapes)

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

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
        box_cls, box_reg, ctr_sco = self.head(features)
        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)

        if self.training:
            while (True):
                standard = "gaussian"
                sigma = 0.5
                mask_thresh = 0.2
                gt_center_masks = []
                # compute center_score
                for im_i in range(len(gt_instances)):
                    gt_per_im = gt_instances[im_i]
                    gt_masks = gt_per_im.gt_masks
                    # standard = 'linear' or 'gaussian', sigma used for gaussian distribution
                    center_masks_i = gt_masks.center_masks(mask_size=gt_per_im.image_size, standard=standard,
                                                           sigma=sigma)
                    gt_center_masks.append(center_masks_i)

                # mask_thresh screen out locations in gt_boxes whose gt_center_score less than threshold
                gt_classes, reg_targets = self.get_ground_truth(locations, gt_instances, gt_center_masks,
                                                                mask_thresh=mask_thresh)

                # computing gt_center_scores and visualization
                valid_ids = gt_classes != self.num_classes
                gt_center_scores = []
                cat_locations = torch.cat(locations, dim=0)
                for im_i in range(len(gt_instances)):
                    valid_id = valid_ids[im_i].nonzero().view(-1)
                    valid_locations = cat_locations[valid_id].long()
                    centers = gt_center_masks[im_i]
                    import ipdb
                    ipdb.set_trace()
                    rgb_i = batched_inputs[im_i]['image'].permute(1, 2, 0).numpy().astype(np.uint8)
                    save(rgb_i, (centers * 255).astype(np.uint8), valid_locations.cpu().numpy(),
                         './train_log/center_masks', 'rgb' + str(im_i) + '.png', 'gaussian_sigma2' + str(im_i) + '.png')
                    centers = torch.Tensor(centers).to(self.device)
                    gt_center_score = centers[
                        valid_locations[:, 1], valid_locations[:, 0]]  # valid_location:xy, centers:[h,w]
                    gt_center_scores.append(gt_center_score)
                gt_center_scores = torch.cat(gt_center_scores, dim=0)
            losses = self.losses(gt_classes, reg_targets, box_cls, box_reg, ctr_sco, gt_center_scores)

            return losses
        else:
            results = self.inference(locations, box_cls, box_reg, ctr_sco, images.image_sizes)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def losses(self, gt_classes, reg_targets, pred_class_logits, pred_box_reg, pred_center_score, gt_center_score):
        pred_class_logits, pred_box_reg, pred_center_score = \
            permute_and_concat(pred_class_logits, pred_box_reg, pred_center_score, self.num_classes)
        # Shapes: (N x R) and (N x R, 4), (N x R) respectively.

        gt_classes = gt_classes.flatten()
        reg_targets = reg_targets.view(-1, 4)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        pos_inds = torch.nonzero(foreground_idxs).squeeze(1)

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        gt_classes_target = torch.zeros_like(pred_class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        cls_loss = sigmoid_focal_loss_jit(
            pred_class_logits, gt_classes_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum",
        ) / num_pos_avg_per_gpu
        if pos_inds.numel() > 0:
            # gt_center_score = compute_centerness_targets(reg_targets[foreground_idxs])

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(gt_center_score.sum()).item() / float(num_gpus)

            reg_loss = iou_loss(
                pred_box_reg[foreground_idxs], reg_targets[foreground_idxs], gt_center_score,
                loss_type=self.iou_loss_type
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                pred_center_score[foreground_idxs], gt_center_score, reduction='sum'
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = pred_box_reg[foreground_idxs].sum()
            reduce_sum(pred_center_score[foreground_idxs].new_tensor([0.0]))
            centerness_loss = pred_center_score[foreground_idxs].sum()

        return dict(cls_loss=cls_loss, reg_loss=reg_loss, centerness_loss=centerness_loss)

    @torch.no_grad()
    def get_ground_truth(self, points, gt_instances, gt_center_masks, mask_thresh):
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
            gt_center_masks, mask_thresh,
            self.fpn_strides, self.center_sampling_radius, self.num_classes, self.norm_reg_targets
        )
        return gt_classes, reg_targets

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
