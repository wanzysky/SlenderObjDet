import math
from typing import List, Optional, Dict

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_every_n
from detectron2.layers import ShapeSpec, cat

from concern.box_ops import box_xyxy_to_cxcywh
from ...layers import Scale, sigmoid_focal_loss
from ..set_head import TransformerSetHead
from ..set_criterion import SetCriterion
from .fcosv2 import (
    get_num_gpus,
    get_sample_region,
    compute_locations,
    compute_targets_for_locations,
    permute_to_N_HW_K,
    reduce_sum
)
from ..matchers import HungarianMatcher


INF = float("inf")


@META_ARCH_REGISTRY.register()
class DeformableParts(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # fmt: off
        self.focal_loss_alpha       = cfg.MODEL.DPM.FOCAL_LOSS_ALPHA
        self.focal_loss_gamma       = cfg.MODEL.DPM.FOCAL_LOSS_GAMMA
        self.smooth_l1_loss_beta    = cfg.MODEL.DPM.SMOOTH_L1_LOSS_BETA
        self.num_classes            = cfg.MODEL.DPM.NUM_CLASSES
        self.in_features            = cfg.MODEL.DPM.IN_FEATURES
        self.fpn_strides            = cfg.MODEL.DPM.FPN_STRIDES
        self.center_sampling_radius = cfg.MODEL.DPM.CENTER_SAMPLING_RADIUS
        self.norm_reg_targets       = cfg.MODEL.DPM.NORM_REG_TARGETS
        # fmt: on

        # Loss parameters:
        giou_weight = cfg.MODEL.DPM.GIOU_WEIGHT
        l1_weight = cfg.MODEL.DPM.L1_WEIGHT
        deep_supervision = cfg.MODEL.DPM.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.DPM.NO_OBJECT_WEIGHT

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        hidden_dim = cfg.MODEL.DPM.HIDDEN_DIM
        d2_backbone = MaskedBackbone(cfg)
        self.backbone = d2_backbone
        self.backbone.num_channels = d2_backbone.num_channels
        backbone_shape = d2_backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.embedding = PartsEmbeddingSine(hidden_dim, normalize=True)
        self.part_head = PartsHead(cfg, feature_shapes)

        self.dense_head = TransformerSetHead(cfg, d2_backbone.num_channels)
        self.size_divisibility = d2_backbone.backbone.size_divisibility

        self.loss_scale = torch.tensor(1e-3)
        matcher = HungarianMatcher(cost_class=1, cost_bbox=l1_weight, cost_giou=giou_weight)
        losses = ["labels", "boxes", "cardinality"]
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        self.criterion = SetCriterion(
            self.num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=no_object_weight, losses=losses
        )
        self.criterion.to(self.device)
        self.to(self.device)

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        nested_features = self.backbone(images)
        features = [nested_features[f].tensors for f in self.in_features]
        masks = [nested_features[f].mask for f in self.in_features]

        part_scores, part_boxes = self.part_head(features)

        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)
        obs_parts = []
        for part, location in zip(part_boxes, locations):
            n, c, h, w = part.shape
            part = part.flatten(2)
            location = location.unsqueeze(0)
            obs_parts.append(
                torch.stack([
                    location[:, :, 0] - part[:, 0],
                    location[:, :, 1] - part[:, 1],
                    location[:, :, 0] + part[:, 2],
                    location[:, :, 1] + part[:, 3],
                    ], dim=1).view(n, c, h, w))

        pos = [self.embedding(m, s.detach(), b.detach())
               for m, s, b in zip(masks, part_scores, obs_parts)]
        pos_indices = [self.select_positives(s) for s in part_scores]

        selected_features = cat([
            f.flatten(2).gather(2, i[:, None].repeat(1, f.shape[1], 1))
            for i, f in zip(pos_indices, features)],
            -1)
        selected_poses = cat([
            f.flatten(2).gather(2, i[:, None].repeat(1, f.shape[1], 1))
            for i, f in zip(pos_indices, pos)],
            -1)
        selected_masks = cat([
            f.flatten(1).gather(1, i)
            for i, f in zip(pos_indices, masks)],
            -1)

        set_pred = self.dense_head(
            selected_features,
            selected_masks,
            selected_poses)
        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_classes, gt_boxes = self.get_ground_truth(locations, gt_instances)

            init_losses = self.losses(
                gt_classes, gt_boxes, locations,
                part_scores, part_boxes)

            targets = self.prepare_targets(gt_instances)
            metrics = self.criterion(set_pred, targets)
            loss_dict = {}
            weight_dict = self.criterion.weight_dict
            momentum = 1e-5
            self.loss_scale = self.loss_scale * (1 - momentum) + 1 * momentum
            for k in list(metrics.keys()):
                if k in weight_dict:
                    loss_dict[k] = metrics[k] * weight_dict[k]
                    loss_dict[k] *= self.loss_scale

            loss_dict.update(init_losses)
            return loss_dict, metrics
        else:
            results = self.inference(part_scores, part_boxes, relation)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)
            gt_classes = targets_per_image.gt_classes
            gt_boxes = targets_per_image.gt_boxes.tensor / image_size_xyxy
            gt_boxes = box_xyxy_to_cxcywh(gt_boxes)
            new_targets.append({"labels": gt_classes, "boxes": gt_boxes})
        return new_targets


    def select_positives(self, scores, positives=512):
        # Declude background
        scores = scores[:, :-1]

        # Pooling and un-pooling for the balance between small and large objects.
        scores, indices = F.max_pool2d(
            scores,
            kernel_size=2, padding=0, stride=2,
            return_indices=True)
        scores = F.max_unpool2d(scores, indices, kernel_size=2, padding=0, stride=2)
        n, c, h, w = scores.shape
        scores, _ = scores.flatten(2).max(1)
        scores, sorted_indices = scores.sort(1)
        sorted_indices = sorted_indices[:, :positives]
        return sorted_indices

    def losses(
            self, gt_classes, gt_boxes, locations,
            pred_scores, pred_boxes):

        def permute_and_concat(box_cls, box_reg, strides, num_classes=80):
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
            strides_flattened = [permute_to_N_HW_K(x, 4) for x in strides]
            # concatenate on the first dimension (representing the feature levels), to
            # take into account the way the labels were generated (with all feature maps
            # being concatenated as well)
            box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
            box_reg = cat(box_reg_flattened, dim=1).view(-1, 4)
            strides = cat(strides_flattened, dim=1).view(-1, 4)

            return box_cls, box_reg, strides

        strides = []
        for i, pred in enumerate(pred_boxes):
            strides.append(
                torch.zeros_like(pred) + self.fpn_strides[i] * 4)

        pred_scores, pred_boxes, strides =\
            permute_and_concat(pred_scores, pred_boxes, strides)

        gt_classes = gt_classes.flatten()
        gt_boxes = gt_boxes.view(-1, 4)

        foreground_idxs = (gt_classes >= 0) & (gt_classes != self.num_classes)
        pos_inds = torch.nonzero(foreground_idxs, as_tuple=False).squeeze(1)

        num_gpus = get_num_gpus()
        total_num_pos = reduce_sum(pos_inds.new_tensor([pos_inds.numel()])).item()
        num_pos_avg_per_gpu = max(total_num_pos / float(num_gpus), 1.0)

        gt_scores = torch.zeros_like(pred_scores)
        gt_scores[foreground_idxs, gt_classes[foreground_idxs]] = 1

        loss_cls = sigmoid_focal_loss_jit(
            pred_scores, gt_scores,
            alpha=self.focal_loss_alpha,
            gamma=self.focal_loss_gamma,
            reduction="sum",
        ) / num_pos_avg_per_gpu

        if pos_inds.numel() == 0:
            loss_box = pred_boxes[foreground_idxs].sum()
            reduce_sum(pred_boxes[foreground_idxs].new_tensor([0.0]))
        else:
            loss_box = smooth_l1_loss(
                pred_boxes[foreground_idxs],
                gt_boxes[foreground_idxs],
                beta=self.smooth_l1_loss_beta,
                reduction="none") / strides[foreground_idxs]
            loss_box, loss_box_indices = loss_box.sort(dim=1)

            loss_box = loss_box[:, :-2].sum() +\
                pred_boxes[foreground_idxs].gather(1, loss_box_indices[:, -2:]).sum() * 1e-3
            loss_box = loss_box / num_pos_avg_per_gpu

        return {"loss_cls": loss_cls * 10,
                "loss_box": loss_box * 10}

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(
                images,
                size_divisibility=self.size_divisibility)
        return images

    @torch.no_grad()
    def get_ground_truth(self, points, gt_instances):
        object_sizes_of_interest = [
            [-1, 64],
            [64, 128],
            [128, 256],
            [256, 512],
            [512, float('inf')],
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
            self.fpn_strides, self.center_sampling_radius, self.num_classes, self.norm_reg_targets
        )
        return gt_classes, reg_targets


class PartsHead(nn.Module):
    """
    Head of part models. Instead of directly predict bboxes of objects,
    part models focus on low-level features and predict local parts.
    """
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super(PartsHead, self).__init__()

        num_classes = cfg.MODEL.DPM.NUM_CLASSES
        in_channels = input_shape[0].channels
        num_convs = cfg.MODEL.DPM.NUM_CONVS

        cls_head = []
        box_head = []
        for _ in range(num_convs):
            cls_head.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_head.append(nn.GroupNorm(32, in_channels))
            cls_head.append(nn.ReLU())

            box_head.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            box_head.append(nn.GroupNorm(32, in_channels))
            box_head.append(nn.ReLU())

        self.logits = nn.Conv2d(in_channels, num_classes, kernel_size=3, stride=1, padding=1)
        self.boxes = nn.Conv2d(in_channels, 4, kernel_size=3, padding=1, stride=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        self.add_module("cls_head", nn.Sequential(*cls_head))
        self.add_module("box_head", nn.Sequential(*box_head))

        # initialization
        for modules in [self.cls_head, self.box_head,
                        self.logits, self.boxes]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.FCOS.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.logits.bias, bias_value)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

    def forward(self, x):
        logits = []
        boxes = []

        for l, feature in enumerate(x):
            logits.append(self.logits(self.cls_head(feature)))
            box_feature = self.box_head(feature)
            box_pred = self.scales[l](self.boxes(box_feature))
            boxes.append(torch.exp(self.scales[l](box_pred)))
        return logits, boxes


class MaskedBackbone(nn.Module):
    """ This is a thin wrapper around D2's backbone to provide padding masking"""

    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg)
        backbone_shape = self.backbone.output_shape()
        self.feature_strides = [backbone_shape[f].stride for f in backbone_shape.keys()]
        self.num_channels = backbone_shape[list(backbone_shape.keys())[-1]].channels

    def forward(self, images):
        features = self.backbone(images.tensor)
        masks = self.mask_out_padding(
            [features_per_level.shape for features_per_level in features.values()],
            images.image_sizes,
            images.tensor.device,
        )
        assert len(features) == len(masks)
        for i, k in enumerate(features.keys()):
            features[k] = NestedTensor(features[k], masks[i])
        return features

    def mask_out_padding(self, feature_shapes, image_sizes, device):
        masks = []
        assert len(feature_shapes) == len(self.feature_strides)
        for idx, shape in enumerate(feature_shapes):
            N, _, H, W = shape
            masks_per_feature_level = torch.ones((N, H, W), dtype=torch.bool, device=device)
            for img_idx, (h, w) in enumerate(image_sizes):
                masks_per_feature_level[
                    img_idx,
                    : int(np.ceil(float(h) / self.feature_strides[idx])),
                    : int(np.ceil(float(w) / self.feature_strides[idx])),
                ] = 0
            masks.append(masks_per_feature_level)
        return masks

    def output_shape(self):
        return self.backbone.output_shape()


class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: Dict[str, NestedTensor] = dict()
        pos = dict()
        for name, x in xs.items():
            out[name] = x
            # position encoding
            pos[name] = self[1](x).to(x.tensors.dtype)

        return out, pos


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, mask, logits, parts):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=parts.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PartsEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None, num_classes=80):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.class_proj = nn.Conv2d(
            num_classes, self.num_pos_feats // 4,
            kernel_size=1, padding=0)

    def forward(self, mask, logits, parts):
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        num_pos_feats = self.num_pos_feats // 4
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=parts.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos_c = self.class_proj(torch.sigmoid(logits)).permute(0, 2, 3, 1)

        num_pos_feats = num_pos_feats // 4
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=parts.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_d = parts[:, :, :, :, None] / dim_t
        pos_d = pos_d.permute((0, 2, 3, 1, 4))
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(3)
        pos = torch.cat(
            (pos_y, pos_x, pos_c, pos_d), dim=3).permute(0, 3, 1, 2)

        return pos
