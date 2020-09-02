from typing import List, Tuple, Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.layers import ShapeSpec, get_norm, cat, batched_nms
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, Instances, ImageList
from slender_det.layers import iou_loss

from .meta_head import HeadBase, MEAT_HEADS_REGISTRY
from .utils import grad_mul, ShiftGenerator, points_to_box
from ..fcos.utils import permute_to_N_HW_K, permute_and_concat_v2, \
    compute_targets_for_locations, compute_centerness_targets, \
    get_num_gpus, reduce_sum


@MEAT_HEADS_REGISTRY.register()
class LRTBHead(HeadBase):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__(cfg, input_shape)
        head_params = cfg.MODEL.META_ARCH

        self.num_points = head_params.NUM_POINTS
        assert self.num_points == 9

        self.fpn_strides = head_params.FPN_STRIDES
        self.center_sampling_radius = head_params.CENTER_SAMPLING_RADIUS
        self.norm_reg_targets = head_params.NORM_REG_TARGETS
        self.centerness_on_loc = head_params.CENTERNESS_ON_LOC

        self.iou_loss_type = head_params.IOU_LOSS_TYPE

        # init bbox pred
        self.loc_init_conv = nn.Conv2d(self.feat_channels, self.loc_feat_channels, 3, 1, 1)
        self.loc_init_out = nn.Conv2d(self.loc_feat_channels, self.num_points * 2, 1, 1, 0)

        # make feature adaptive layer
        self.cls_conv, self.loc_refine_conv = self.make_featre_adaptive_layers()
        self._make_offset()

        self.cls_out = nn.Conv2d(self.feat_channels, self.num_classes, 1, 1, 0)
        self.ctn_out = nn.Conv2d(self.feat_channels, 1, 1, 1, 0)
        self.loc_refine_out = nn.Conv2d(self.loc_feat_channels, self.num_points * 2, 1, 1, 0)

        self._init_weights()

    def _prepare_offset(self):
        if self.feat_adaptive == "Unsupervised Offset":
            self.offset_conv = nn.Conv2d(self.feat_channels, 3 * 3 * 2, 1, 1, 0)
        elif self.feat_adaptive == "Supervised Offset":
            self.dcn_kernel = int(np.sqrt(self.num_points))
            self.dcn_pad = int((self.dcn_kernel - 1) / 2)
            dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
            dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
            dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
            dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
            self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)

    def _init_weights(self):
        # Initialization
        for modules in [
            self.loc_init_conv, self.loc_init_out, self.cls_conv,
            self.loc_refine_conv, self.cls_out, self.loc_refine_out
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)

        # Use prior in model initialization to improve stability
        bias_value = -(math.log((1 - self.prior_prob) / self.prior_prob))
        nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            gt_instances: Optional[List[Instances]] = None
    ):
        cls_outs, ctn_outs, loc_outs_init, loc_outs_refine = self._forward(features)
        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)

        if self.training:
            return self.losses(locations, cls_outs, ctn_outs, loc_outs_init, loc_outs_refine, gt_instances)
        else:
            results = self.inference(center_pts, cls_outs, ctn_outs, loc_outs_init, loc_outs_refine, images)
            processed_results = []
            for results_per_image, input_per_image, image_size in zip(
                    results, batched_inputs, images.image_sizes):
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                r = detector_postprocess(results_per_image, height, width)
                processed_results.append({"instances": r})
            return processed_results

    def _forward(self, features):
        cls_outs = []
        ctn_outs = []
        loc_outs_init = []
        loc_outs_refine = []

        dcn_base_offsets = self.dcn_base_offset.type_as(features[0])
        for l, feature in enumerate(features):
            cls_feat = feature
            loc_feat = feature

            for cls_conv in self.cls_subnet:
                cls_feat = cls_conv(cls_feat)
            for loc_conv in self.loc_subnet:
                loc_feat = loc_conv(loc_feat)

            loc_out_init = self.loc_init_out(F.relu_(self.loc_init_conv(loc_feat)))
            loc_outs_init.append(loc_out_init[:, :4, :, :])

            if self.feat_adaption == "None":
                cls_feat_fa = self.cls_conv(cls_feat)
                loc_feat_fa = self.loc_refine_conv(loc_feat)
            elif self.feat_adaption == "Unsupervised Offset":
                # TODO: choose a better input info for generating offsets
                dcn_offsets = self.offset_conv(loc_feat)
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets)
            elif self.feat_adaption == "Supervised Offset":
                # build offsets for deformable conv
                loc_out_init_grad_mul = grad_mul(loc_out_init, self.gradient_mul)
                # TODOs: commpute offset for different methods
                dcn_offsets = loc_out_init_grad_mul - dcn_base_offsets
                # get adaptive feature map
                cls_feat_fa = self.cls_conv(cls_feat, dcn_offsets)
                loc_feat_fa = self.loc_refine_conv(loc_feat, dcn_offsets)
            else:
                raise RuntimeError("Got {}".format(self.feat_adaption))

            cls_outs.append(self.cls_out(F.relu_(cls_feat_fa)))
            ctn_outs.append(self.ctn_out(F.relu_(cls_feat_fa)))

            if self.res_refine:
                loc_out_refine = self.reg_refine_out(F.relu_(loc_feat_fa)) + loc_out_init.detach()
            else:
                loc_out_refine = self.reg_refine_out(F.relu_(loc_feat_fa))

            loc_outs_refine.append(loc_out_refine[:, :4, :, :])

        return cls_outs, ctn_outs, reg_outs_init, reg_outs_refine

    def losses(self, locations, class_logits, center_score, box_reg_init, box_reg, gt_instances):
        gt_classes, loc_targets = self.get_ground_truth(locations, gt_instances)

        class_logits, box_reg_init, box_reg, center_score = permute_and_concat_v2(
            class_logits, box_reg_init, box_reg, center_score, self.num_classes)
        # Shapes: (N x R) and (N x R, 4), (N x R) respectively.

        gt_classes = gt_classes.flatten()
        loc_targets = loc_targets.view(-1, 4)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        pos_inds = torch.nonzero(foreground_idxs).squeeze(1)

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        gt_classes_target = torch.zeros_like(class_logits)
        gt_classes_target[foreground_idxs, gt_classes[foreground_idxs]] = 1

        # logits loss
        cls_loss = sigmoid_focal_loss_jit(
            class_logits, gt_classes_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum",
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            gt_center_score = compute_centerness_targets(loc_targets[foreground_idxs])
            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(gt_center_score.sum()).item() / float(num_gpus)

            loss_loc_init = iou_loss(
                box_reg[foreground_idxs], loc_targets[foreground_idxs], gt_center_score,
                loss_type=self.iou_loss_type
            ) / sum_centerness_targets_avg_per_gpu * self.loss_bbox_init_weight

            loss_loc_refine = iou_loss(
                box_reg[foreground_idxs], loc_targets[foreground_idxs], gt_center_score,
                loss_type=self.iou_loss_type
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                center_score[foreground_idxs], gt_center_score, reduction='sum'
            ) / num_pos_avg_per_gpu
        else:
            loss_loc_init = box_reg[foreground_idxs].sum()
            loss_loc_refine = box_reg[foreground_idxs].sum()
            reduce_sum(center_score[foreground_idxs].new_tensor([0.0]))
            centerness_loss = center_score[foreground_idxs].sum()

        return dict(
            loss_cls=cls_loss * self.loss_cls_weight,
            centerness_loss=centerness_loss * self.loss_cls_weight,
            loss_loc_init=loss_loc_init * self.loss_loc_init_weight,
            loss_loc_refine=loss_loc_refine * self.loss_loc_refine_weight,
        )

    @torch.no_grad()
    def get_ground_truth(self, points, gt_instances):
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

        gt_classes, loc_targets = compute_targets_for_locations(
            points, gt_instances, expanded_object_sizes_of_interest,
            self.fpn_strides, self.center_sampling_radius, self.num_classes
        )
        return gt_classes, loc_targets

    def inference(self, locations, box_cls, ctr_sco, box_reg_init, box_reg, image_sizes):
        results = []

        box_cls = [permute_to_N_HW_K(x, self.num_classes) for x in box_cls]
        box_reg = [permute_to_N_HW_K(x, 4) for x in box_reg]
        ctr_sco = [permute_to_N_HW_K(x, 1) for x in ctr_sco]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_init_per_image = [box_reg_init_per_level[img_idx] for box_reg_init_per_level in box_reg_init]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_reg]
            ctr_sco_per_image = [ctr_sco_per_level[img_idx] for ctr_sco_per_level in ctr_sco]

            results_per_image = self.inference_single_image(
                locations, box_cls_per_image, box_reg_init_per_image,
                box_reg_per_image, ctr_sco_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

    def inference_single_image(self, locations, box_cls, box_reg_init, box_reg, center_score, image_size):
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