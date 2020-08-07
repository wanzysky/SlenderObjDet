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
from detectron2.layers import ShapeSpec, batched_nms
from detectron2.modeling.postprocessing import detector_postprocess

from ..backbone import build_backbone
from ...layers import Scale, iou_loss, DFConv2d

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


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
        locations, targets, object_sizes_of_interest,
        strides, num_points_per_level, center_sampling_radius, num_classes
):
    labels = []
    reg_targets = []
    xs, ys = locations[:, 0], locations[:, 1]

    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        bboxes = targets_per_im.gt_boxes.tensor
        labels_per_im = targets_per_im.gt_classes
        area = targets_per_im.gt_boxes.area()

        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        if center_sampling_radius > 0:
            is_in_boxes = get_sample_region(
                bboxes, strides, num_points_per_level,
                xs, ys, radius=center_sampling_radius
            )
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

        reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]
        labels_per_im = labels_per_im[locations_to_gt_inds]
        # NOTE: set background labels to NUM_CLASSES not 0
        labels_per_im[locations_to_min_area == INF] = num_classes

        labels.append(labels_per_im)
        reg_targets.append(reg_targets_per_im)

    return labels, reg_targets


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


@META_ARCH_REGISTRY.register()
class FCOS(nn.Module):

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
        box_cls, box_regression, centerness = self.head(features)
        # compute ground truth location (x, y)
        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)

        if self.training:
            labels, reg_targets = self.get_ground_truth(locations, gt_instances)
            losses = self.losses(labels, reg_targets, box_cls, box_regression, centerness)

            return losses
        else:
            results = self.inference(locations, box_cls, box_regression, centerness, images.image_sizes)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def losses(self, labels, reg_targets, box_cls, box_regression, centerness):
        N, num_classes = box_cls[0].shape[:2]

        box_cls_flatten = []
        box_regression_flatten = []
        centerness_flatten = []
        labels_flatten = []
        reg_targets_flatten = []
        for l in range(len(labels)):
            box_cls_flatten.append(box_cls[l].permute(0, 2, 3, 1).reshape(-1, num_classes))
            box_regression_flatten.append(box_regression[l].permute(0, 2, 3, 1).reshape(-1, 4))
            labels_flatten.append(labels[l].reshape(-1))
            reg_targets_flatten.append(reg_targets[l].reshape(-1, 4))
            centerness_flatten.append(centerness[l].reshape(-1))

        box_cls_flatten = torch.cat(box_cls_flatten, dim=0)
        box_regression_flatten = torch.cat(box_regression_flatten, dim=0)
        centerness_flatten = torch.cat(centerness_flatten, dim=0)
        labels_flatten = torch.cat(labels_flatten, dim=0)
        reg_targets_flatten = torch.cat(reg_targets_flatten, dim=0)

        pos_inds = torch.nonzero(
            (labels_flatten >= 0) & (labels_flatten != self.num_classes)
        ).squeeze(1)

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]

        num_gpus = get_num_gpus()
        # sync num_pos from all gpus
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        gt_classes_target = torch.zeros_like(box_cls_flatten)
        foreground_idxs = (labels_flatten >= 0) & (labels_flatten != self.num_classes)
        gt_classes_target[foreground_idxs, labels_flatten[foreground_idxs]] = 1

        cls_loss = sigmoid_focal_loss_jit(
            box_cls_flatten, gt_classes_target,
            alpha=self.focal_loss_alpha, gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() > 0:
            centerness_targets = compute_centerness_targets(reg_targets_flatten)
            # average sum_centerness_targets from all gpus,
            # which is used to normalize centerness-weighed reg loss
            sum_centerness_targets_avg_per_gpu = \
                reduce_sum(centerness_targets.sum()).item() / float(num_gpus)

            reg_loss = iou_loss(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets,
                loss_type=self.iou_loss_type
            ) / sum_centerness_targets_avg_per_gpu

            centerness_loss = F.binary_cross_entropy_with_logits(
                centerness_flatten,
                centerness_targets,
                reduction='sum'
            ) / num_pos_avg_per_gpu
        else:
            reg_loss = box_regression_flatten.sum()
            reduce_sum(centerness_flatten.new_tensor([0.0]))
            centerness_loss = centerness_flatten.sum()

        return dict(cls_loss=cls_loss, reg_loss=reg_loss, centerness_loss=centerness_loss)

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
        num_points_per_level = [len(points_per_level) for points_per_level in points]
        self.num_points_per_level = num_points_per_level
        points_all_level = torch.cat(points, dim=0)
        labels, reg_targets = compute_targets_for_locations(
            points_all_level, gt_instances,
            expanded_object_sizes_of_interest,
            self.fpn_strides, self.num_points_per_level,
            self.center_sampling_radius, self.num_classes
        )
        for i in range(len(labels)):
            labels[i] = torch.split(labels[i], num_points_per_level, dim=0)
            reg_targets[i] = torch.split(reg_targets[i], num_points_per_level, dim=0)

        labels_level_first = []
        reg_targets_level_first = []
        for level in range(len(points)):
            labels_level_first.append(
                torch.cat([labels_per_im[level] for labels_per_im in labels], dim=0)
            )

            reg_targets_per_level = torch.cat([
                reg_targets_per_im[level]
                for reg_targets_per_im in reg_targets
            ], dim=0)

            if self.norm_reg_targets:
                reg_targets_per_level = reg_targets_per_level / self.fpn_strides[level]
            reg_targets_level_first.append(reg_targets_per_level)

        return labels_level_first, reg_targets_level_first

    def inference(self, locations, box_cls, box_regression, centerness, image_sizes):
        results = []
        for _, (l, o, b, c) in enumerate(zip(locations, box_cls, box_regression, centerness)):
            results.append(
                self.inference_single_feature_map(l, o, b, c, image_sizes)
            )

        results = list(zip(*results))
        results = [Instances.cat(_) for _ in results]
        return results

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

            result = Instances(image_sizes[i])
            result.pred_boxes = Boxes(detections)
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


class FCOSHead(torch.nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSHead, self).__init__()
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
        self.cls_logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.bbox_pred = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.centerness = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)

        # initialization
        for modules in [self.cls_tower, self.bbox_tower,
                        self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        bbox_reg = []
        centerness = []
        for l, feature in enumerate(x):
            cls_tower = self.cls_tower(feature)
            box_tower = self.bbox_tower(feature)

            logits.append(self.cls_logits(cls_tower))
            if self.centerness_on_reg:
                centerness.append(self.centerness(box_tower))
            else:
                centerness.append(self.centerness(cls_tower))

            bbox_pred = self.scales[l](self.bbox_pred(box_tower))
            if self.norm_reg_targets:
                bbox_pred = F.relu(bbox_pred)
                if self.training:
                    bbox_reg.append(bbox_pred)
                else:
                    bbox_reg.append(bbox_pred * self.fpn_strides[l])
            else:
                bbox_reg.append(torch.exp(bbox_pred))
        return logits, bbox_reg, centerness
