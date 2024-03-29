import os
import math
from typing import List

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
from slender_det.layers import Scale, iou_loss, DFConv2d, anchor_iou_loss

from .utils import INF, get_num_gpus, reduce_sum, permute_to_N_HW_K, \
    compute_locations_per_level, compute_locations, get_sample_region, \
    compute_centerness_targets, compute_targets_for_locations

from detectron2.modeling.anchor_generator import build_anchor_generator


def softmax(x):
    x_exp = torch.exp(x)
    x_sum = torch.sum(x_exp, 1)
    return x_exp/x_sum[:,None]
    
    
def get_anchor_offsets(locations, anchors, num_anchors):
    anchors_tensor = [anchors[i].tensor for i in range(len(anchors))]
    anchors_offsets = []
    for i, loc_i in enumerate(locations):
        anchors_i = anchors_tensor[i]
        loc_i = loc_i.repeat(1,num_anchors).view(-1,2)
        lt = loc_i - anchors_i[:,[0,1]]
        rb = anchors_i[:,[2,3]] - loc_i
        anchor_offset = torch.cat((lt,rb), dim=1)
        anchors_offsets.append(anchor_offset)
    return anchors_offsets
    
    
def permute_and_concat(box_cls, box_reg, center_score, shape_cls, num_classes=80, num_anchors = 9):
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
    center_score = [permute_to_N_HW_K(x, 1) for x in center_score]
    shape_cls = [permute_to_N_HW_K(x, num_anchors) for x in shape_cls]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_reg = cat(box_reg_flattened, dim=1).view(-1, 4)
    center_score = cat(center_score, dim=1).view(-1)
    shape_cls = cat(shape_cls, dim=1).view(-1, num_anchors)

    return box_cls, box_reg, center_score, shape_cls
    
    
@META_ARCH_REGISTRY.register()
class FCOSAnchor(nn.Module):

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
        self.head = FCOSAnchorHead(cfg, feature_shapes)
        self.anchor_generator = build_anchor_generator(cfg, feature_shapes)
        self.num_anchors = self.anchor_generator.num_cell_anchors[0]

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
        anchors = self.anchor_generator(features)
        
        box_cls, box_reg, ctr_sco, shape_cls = self.head(features)

        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)
        
        anchor_offsets = get_anchor_offsets(locations, anchors, self.num_anchors)
        anchor_targets = []
        for im_i in range(len(gt_instances)):
            anchor_offsets_cat = torch.cat(anchor_offsets, dim = 0)
            anchor_targets.append(anchor_offsets_cat)
        anchor_targets = torch.stack(anchor_targets)
            
        if self.training:
            gt_classes, reg_targets = self.get_ground_truth(locations, gt_instances)
            losses = self.losses(gt_classes, reg_targets, anchor_targets, box_cls, box_reg, ctr_sco, shape_cls)

            return losses
        else:
            results = self.inference(locations, anchor_offsets, box_cls, box_reg, ctr_sco, shape_cls, images.image_sizes)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def losses(self, gt_classes, reg_targets, anchor_targets, pred_class_logits, pred_box_reg, pred_center_score, pred_shape_logits):
        pred_class_logits, pred_box_reg, pred_center_score, pred_shape_logits = \
            permute_and_concat(pred_class_logits, pred_box_reg, pred_center_score, pred_shape_logits, self.num_classes)
        # Shapes: (N x R) and (N x R, 4), (N x R) respectively.

        gt_classes = gt_classes.flatten()
        reg_targets = reg_targets.view(-1, 4)
        anchor_targets = anchor_targets.view(-1,4)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        pos_inds = torch.nonzero(foreground_idxs).squeeze(1)
        
        num_anchors = len(anchor_targets)//len(reg_targets)
        expand_reg_targets = reg_targets.repeat(1,num_anchors).view(-1,4)
        expand_foreground_idx = foreground_idxs[:,None].repeat(1,num_anchors).view(-1)
        expand_pos_inds = torch.nonzero(expand_foreground_idx).squeeze(1)
        pred_shape_logits = softmax(pred_shape_logits).view(-1)
        
        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)
        
        expand_total_num_pos = reduce_sum(expand_pos_inds.new_tensor([expand_pos_inds.numel()])).item()
        expand_num_pos_avg_per_gpu = max(expand_total_num_pos / float(num_gpus), 1.0)

        gt_classes_target = torch.zeros_like(pred_class_logits)
        foreground_gt_classes = gt_classes[foreground_idxs][:,None].repeat(1,num_anchors).view(-1)
        gt_classes_target[expand_foreground_idx, foreground_gt_classes] = 1

        # logits loss
        cls_loss = sigmoid_focal_loss_jit(
            pred_class_logits, gt_classes_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma, reduction="sum",
        ) / expand_num_pos_avg_per_gpu
        if pos_inds.numel() > 0:
            gt_center_score = compute_centerness_targets(reg_targets[foreground_idxs])
            gt_center_score_for_anchors = gt_center_score[:,None].repeat(1,num_anchors).view(-1)\
                * pred_shape_logits[expand_foreground_idx]

            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(gt_center_score_for_anchors.sum()).item() / float(num_gpus)

            norm_loss, reg_loss = anchor_iou_loss(
                pred_box_reg[expand_foreground_idx]+anchor_targets[expand_foreground_idx], \
                    expand_reg_targets[expand_foreground_idx], gt_center_score_for_anchors,
                loss_type=self.iou_loss_type
            ) 
            reg_loss = reg_loss / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                pred_center_score[foreground_idxs], gt_center_score, reduction='sum'
            ) / num_pos_avg_per_gpu
            
            shape_loss = F.binary_cross_entropy_with_logits(
                pred_shape_logits[expand_foreground_idx], norm_loss, reduction='sum'
            ) / expand_num_pos_avg_per_gpu
        else:
            reg_loss = pred_box_reg[expand_foreground_idx].sum()
            reduce_sum(pred_center_score[foreground_idxs].new_tensor([0.0]))
            centerness_loss = pred_center_score[foreground_idxs].sum()
            shape_loss = pred_shape_logits[expand_foreground_idx].sum()

        return dict(cls_loss=cls_loss, reg_loss=reg_loss, centerness_loss=centerness_loss, shape_loss = shape_loss)

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

        gt_classes, reg_targets = compute_targets_for_locations(
            points, gt_instances, expanded_object_sizes_of_interest,
            self.fpn_strides, self.center_sampling_radius, self.num_classes
        )
        return gt_classes, reg_targets

    def inference(self, locations, anchor_offsets, box_cls, box_reg, ctr_sco, shape_cls, image_sizes):
        results = []

        box_cls = [permute_to_N_HW_K(x, self.num_classes) for x in box_cls]
        box_reg = [permute_to_N_HW_K(x, 4) for x in box_reg]
        ctr_sco = [permute_to_N_HW_K(x, 1) for x in ctr_sco]
        shape_cls = [permute_to_N_HW_K(x, 1) for x in shape_cls]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_reg]
            ctr_sco_per_image = [ctr_sco_per_level[img_idx] for ctr_sco_per_level in ctr_sco]
            shape_cls_per_image = [shape_cls_per_level[img_idx] for shape_cls_per_level in shape_cls]
            
            results_per_image = self.inference_single_image(
                locations, anchor_offsets, box_cls_per_image, box_reg_per_image, ctr_sco_per_image, shape_cls_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

    def inference_single_image(self, locations, anchor_targets, box_cls, box_reg, center_score, shape_cls, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []
        
        # Iterate over every feature level
        for box_cls_i, box_reg_i, locs_i, anchor_offset_i, center_score_i, shape_cls_i \
            in zip(box_cls, box_reg, locations, anchor_offsets, center_score, shape_cls):
            
            center_score_i = center_score_i[:,None].repeat(1,self.num_anchors).view(-1)
            locs_i = locs_i.repeat(1,self.num_anchors).view(-1,2)
            # (HxW, C)
            box_cls_i = box_cls_i.sigmoid_()
            keep_idxs = box_cls_i > self.pre_nms_thresh

            # multiply the classification scores with center scores
            box_cls_i *= (center_score_i*shape_cls_i).sigmoid_()
            
            box_cls_i = box_cls_i[keep_idxs]
            keep_idxs_nonzero_i = keep_idxs.nonzero()

            box_loc_i = keep_idxs_nonzero_i[:, 0]
            class_i = keep_idxs_nonzero_i[:, 1]

            box_reg_i = box_reg_i[box_loc_i]
            locs_i = locs_i[box_loc_i]
            anchor_offset_i = anchor_offset_i[box_loc_i]

            per_pre_nms_top_n = keep_idxs.sum().clamp(max=self.pre_nms_top_n)
            if keep_idxs.sum().item() > per_pre_nms_top_n.item():
                box_cls_i, topk_idxs = box_cls_i.topk(per_pre_nms_top_n, sorted=False)

                class_i = class_i[topk_idxs]
                box_reg_i = box_reg_i[topk_idxs]
                locs_i = locs_i[topk_idxs]
                anchor_offset_i = anchor_offset_i[topk_idxs]

            # predict boxes
            predicted_boxes = torch.stack([
                locs_i[:, 0] - box_reg_i[:, 0] - anchor_offset_i[:,0], locs_i[:, 1] - box_reg_i[:, 1] - anchor_offset_i[:,1],
                locs_i[:, 0] + box_reg_i[:, 2] + anchor_offset_i[:,2], locs_i[:, 1] + box_reg_i[:, 3] + anchor_offset_i[:,3],
            ], dim=1)
            #predicted_boxes = locs_i + box_reg_i
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


class FCOSAnchorHead(torch.nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSAnchorHead, self).__init__()
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
        
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors[0]
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
        self.cls_logits = nn.Conv2d(in_channels, num_classes * num_anchors, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4 * num_anchors, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        self.shape_logits = nn.Conv2d(in_channels, num_anchors, kernel_size=3, stride=1, padding=1)
        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_logits, self.bbox_pred, self.centerness, self.shape_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
                # add weight init for gn
                if isinstance(l, nn.GroupNorm):
                    torch.nn.init.constant_(l.weight, 1)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)
        torch.nn.init.constant_(self.shape_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        shape_logits = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            shape_logits.append(self.shape_logits(cls_tower).sigmoid_())
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_reg.append(F.relu(bbox_pred) * self.fpn_strides[l])
            else:
                #bbox_reg.append(torch.exp(bbox_pred))
                bbox_reg.append(bbox_pred)
        return logits, bbox_reg, centerness, shape_logits

