import os
import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.modeling.postprocessing import detector_postprocess

from ..backbone import build_backbone
from ...layers import Scale, DFConv2d

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def permute_to_N_HW_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, C, H, W) to (N, (HxW), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.permute(0, 2, 3, 1).reshape(N, -1, K)  # Size=(N, HxW, K)
    return tensor


def permute_and_concat(box_cls, box_delta, center_score, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_delta and the center-ness
    box_cls_flattened = [permute_to_N_HW_K(x, num_classes) for x in box_cls]
    box_delta_flattened = [permute_to_N_HW_K(x, 4) for x in box_delta]
    center_score = [permute_to_N_HW_K(x, 1) for x in center_score]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_delta = cat(box_delta_flattened, dim=1).view(-1, 4)
    center_score = cat(center_score, dim=1).view(-1)

    return box_cls, box_delta, center_score


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_locations(shapes, strides, device):
    locations = []
    for level, (shape, stride) in enumerate(zip(shapes, strides)):
        h, w = shape
        locations_per_level = compute_locations_per_level(h, w, stride, device)
        locations.append(locations_per_level)

    return locations


def get_sample_region(gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
    """
    This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
    """
    num_gts = gt.shape[0]
    K = len(gt_xs)
    gt = gt[None].expand(K, num_gts, 4)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    center_gt = gt.new_zeros(gt.shape)

    # no gt
    if center_x[..., 0].sum() == 0:
        return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

    beg = 0
    for level, n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
        center_gt[beg:end, :, 0] = torch.where(
            xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
        )
        center_gt[beg:end, :, 1] = torch.where(
            ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
        )
        center_gt[beg:end, :, 2] = torch.where(
            xmax > gt[beg:end, :, 2],
            gt[beg:end, :, 2], xmax
        )
        center_gt[beg:end, :, 3] = torch.where(
            ymax > gt[beg:end, :, 3],
            gt[beg:end, :, 3], ymax
        )
        beg = end

    left = gt_xs[:, None] - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs[:, None]
    top = gt_ys[:, None] - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys[:, None]
    center_bbox = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

    return inside_gt_bbox_mask


def compute_targets_for_locations(
        locations, anchors, box2box_transform: Box2BoxTransform, targets, object_sizes_of_interest,
        strides, center_sampling_radius, num_classes, norm_reg_targets=False
):
    assert len(locations) == len(anchors), "Number of locations must be equal to the number of anchors," \
                                           "but got {} locations and {} anchors".format(len(locations), len(anchors))

    num_points = [len(_) for _ in locations]
    # build normalization weights before cat locations
    norm_weights = None
    if norm_reg_targets:
        norm_weights = torch.cat([torch.empty(n).fill_(s) for n, s in zip(num_points, strides)])

    locations = torch.cat(locations, dim=0)
    xs, ys = locations[:, 0], locations[:, 1]
    anchors = Boxes.cat(anchors)  # Rx4

    gt_classes = []
    gt_anchors_deltas = []
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
        reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
        if norm_reg_targets and norm_weights is not None:
            reg_targets_per_im /= norm_weights[:, None]

        # generate box deltas
        matched_gt_boxes = targets_per_im.gt_boxes[locations_to_gt_inds]
        gt_anchors_reg_deltas_per_im = box2box_transform.get_deltas(anchors.tensor, matched_gt_boxes.tensor)

        gt_classes.append(gt_classes_per_im)
        reg_targets.append(reg_targets_per_im)
        gt_anchors_deltas.append(gt_anchors_reg_deltas_per_im)

    return torch.stack(gt_classes), torch.stack(reg_targets), torch.stack(gt_anchors_deltas)


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


@META_ARCH_REGISTRY.register()
class FCOSAnchor(nn.Module):

    def __init__(self, cfg):
        super().__init__()

        self.in_features = cfg.MODEL.FCOS.IN_FEATURES

        # Loss parameters:
        # defined by method<get_ground_truth>
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.FCOS.CENTER_SAMPLING_RADIUS
        self.norm_reg_targets = cfg.MODEL.FCOS.NORM_REG_TARGETS

        self.focal_loss_alpha = cfg.MODEL.FCOS.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma = cfg.MODEL.FCOS.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta = cfg.MODEL.FCOS.SMOOTH_L1_LOSS_BETA

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

        self.box2box_transform = Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS)

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
        box_cls, box_delta, center_score = self.head(features)

        # compute ground truth location (x, y) and ground truth anchors (x0, y0, x1, y1)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)
        anchors = self.anchor_generator(features)

        if self.training:
            gt_classes, reg_targets, gt_anchors_deltas = self.get_ground_truth(locations, anchors, gt_instances)
            losses = self.losses(gt_classes, reg_targets, gt_anchors_deltas, box_cls, box_delta, center_score)

            return losses
        else:
            results = self.inference(anchors, box_cls, box_delta, center_score, images.image_sizes)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def losses(
            self, gt_classes, reg_targets, gt_anchors_deltas,
            pred_class_logits, pred_anchor_deltas, pred_center_score
    ):
        pred_class_logits, pred_anchor_deltas, pred_center_score = \
            permute_and_concat(pred_class_logits, pred_anchor_deltas, pred_center_score, self.num_classes)
        # Shapes: (N x R) and (N x R, 4), (N x R) respectively.

        gt_classes = gt_classes.flatten()
        reg_targets = reg_targets.view(-1, 4)
        gt_anchors_deltas = gt_anchors_deltas.view(-1, 4)

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
            reg_loss = smooth_l1_loss(
                pred_anchor_deltas[foreground_idxs], gt_anchors_deltas[foreground_idxs],
                beta=self.smooth_l1_loss_beta, reduction="sum",
            ) / num_pos_avg_per_gpu

            gt_center_score = compute_centerness_targets(reg_targets[foreground_idxs])
            centerness_loss = F.binary_cross_entropy_with_logits(
                pred_center_score[foreground_idxs], gt_center_score, reduction='sum'
            ) / num_pos_avg_per_gpu

        else:
            reg_loss = pred_anchor_deltas[foreground_idxs].sum()
            reduce_sum(pred_center_score[foreground_idxs].new_tensor([0.0]))
            centerness_loss = pred_center_score[foreground_idxs].sum()

        return dict(cls_loss=cls_loss, reg_loss=reg_loss, centerness_loss=centerness_loss)

    @torch.no_grad()
    def get_ground_truth(self, points, anchors, gt_instances):
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

        gt_classes, reg_targets, gt_anchors_deltas = compute_targets_for_locations(
            points, anchors, self.box2box_transform, gt_instances,
            expanded_object_sizes_of_interest,
            self.fpn_strides, self.center_sampling_radius, self.num_classes
        )
        return gt_classes, reg_targets, gt_anchors_deltas

    def inference(self, anchors, box_cls, box_deltas, ctr_sco, image_sizes):
        results = []

        box_cls = [permute_to_N_HW_K(x, self.num_classes) for x in box_cls]
        box_deltas = [permute_to_N_HW_K(x, 4) for x in box_deltas]
        ctr_sco = [permute_to_N_HW_K(x, 1) for x in ctr_sco]
        # list[Tensor], one per level, each has shape (N, Hi x Wi x A, K or 4)

        for img_idx, image_size in enumerate(image_sizes):
            box_cls_per_image = [box_cls_per_level[img_idx] for box_cls_per_level in box_cls]
            box_reg_per_image = [box_reg_per_level[img_idx] for box_reg_per_level in box_deltas]
            ctr_sco_per_image = [ctr_sco_per_level[img_idx] for ctr_sco_per_level in ctr_sco]

            results_per_image = self.inference_single_image(
                anchors, box_cls_per_image, box_reg_per_image, ctr_sco_per_image, tuple(image_size)
            )
            results.append(results_per_image)

        return results

    def inference_single_image(self, anchors, box_cls, box_delta, center_score, image_size):
        boxes_all = []
        scores_all = []
        class_idxs_all = []

        # Iterate over every feature level
        for box_cls_i, box_reg_i, anchors_i, center_score_i in zip(box_cls, box_delta, anchors, center_score):
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
            anchors_i = anchors_i[box_loc_i]

            per_pre_nms_top_n = keep_idxs.sum().clamp(max=self.pre_nms_top_n)
            if keep_idxs.sum().item() > per_pre_nms_top_n.item():
                box_cls_i, topk_idxs = box_cls_i.topk(per_pre_nms_top_n, sorted=False)

                class_i = class_i[topk_idxs]
                box_reg_i = box_reg_i[topk_idxs]
                anchors_i = anchors_i[topk_idxs]

            # predict boxes
            predicted_boxes = self.box2box_transform.apply_deltas(box_reg_i, anchors_i.tensor)
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

        num_convs = cfg.MODEL.FCOS.NUM_CONVS
        num_anchors = build_anchor_generator(cfg, input_shape).num_cell_anchors
        # fmt: on
        assert (
                len(set(num_anchors)) == 1
        ), "Using different number of anchors between levels is not currently supported!"
        num_anchors = num_anchors[0]
        assert (num_anchors == 1), "FCOS Anchor can only have one anchor per scale!"

        cls_tower = []
        bbox_tower = []
        for i in range(num_convs):
            use_dcn = False
            use_v2 = True
            if self.use_dcn_in_tower and i == num_convs - 1:
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
        self.cls_score = nn.Conv2d(in_channels, num_anchors * num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower, self.cls_score, self.bbox_pred, self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_score.bias, bias_value)

    def forward(self, x):
        logits = []
        bbox_reg = []
        center_score = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_score(cls_tower))
            bbox_reg.append(self.bbox_pred(box_tower))
            if self.centerness_on_reg:
                center_score.append(self.centerness(box_tower))
            else:
                center_score.append(self.centerness(cls_tower))

        return logits, bbox_reg, center_score
