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
from detectron2.layers import ShapeSpec, cat

from ...layers import Scale
from ..non_local_head import TransformerNonLocal, Conv2dNonLocal
from .fcosv2 import (
    get_num_gpus,
    get_sample_region,
    compute_locations,
    compute_targets_for_locations,
    permute_to_N_HW_K,
    reduce_sum
)


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

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        hidden_dim = cfg.MODEL.DPM.HIDDEN_DIM
        N_steps = hidden_dim // 2
        d2_backbone = MaskedBackbone(cfg)
        self.backbone = d2_backbone
        self.backbone.num_channels = d2_backbone.num_channels
        backbone_shape = d2_backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in self.in_features]
        self.embedding = PartsEmbeddingSine(N_steps, normalize=True)
        self.part_head = PartsHead(cfg, feature_shapes)
        self.dense_head = Conv2dNonLocal(cfg, d2_backbone.num_channels)
        self.size_divisibility = d2_backbone.backbone.size_divisibility
        self.loss_nominator = torch.tensor(1e-3)
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

        part_scores, part_boxes, part_confidences = self.part_head(features)

        pos = [self.embedding(m, s, b)
               for m, s, b in zip(masks, part_scores, part_boxes)]
        relation = self.dense_head(features, masks, pos)

        shapes = [feature.shape[-2:] for feature in features]
        locations = compute_locations(shapes, self.fpn_strides, self.device)

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            gt_classes, gt_boxes = self.get_ground_truth(locations, gt_instances)

            losses = self.losses(
                gt_classes, gt_boxes, locations,
                part_scores, part_boxes, part_confidences, relation)
            return losses
        else:
            results = self.inference(part_scores, part_boxes, relation)
            results = self.postprocess(results, batched_inputs, images.image_sizes)

            return results

    def delta2boxes(self, location, delta):
        pass


    def losses(
        self, gt_classes, gt_boxes, locations,
        pred_scores, pred_boxes, pred_confidences, pred_relations):

        def permute_and_concat(box_cls, box_reg, confidences, strides, num_classes=80):
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
            confidence_flattened = [permute_to_N_HW_K(x, 4) for x in confidences]
            strides_flattened = [permute_to_N_HW_K(x, 4) for x in strides]
            # concatenate on the first dimension (representing the feature levels), to
            # take into account the way the labels were generated (with all feature maps
            # being concatenated as well)
            box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
            box_reg = cat(box_reg_flattened, dim=1).view(-1, 4)
            confidences = cat(confidence_flattened, dim=1).view(-1, 4)
            strides = cat(strides_flattened, dim=1).view(-1, 4)

            return box_cls, box_reg, confidences, strides

        strides = []
        connected_boxes = []
        for i, pred in enumerate(pred_boxes):
            strides.append(
                torch.zeros_like(pred) + self.fpn_strides[i] * 4)
            n, c, h, w = pred.shape
            init_pred = pred
            pred = F.max_pool2d(pred, 2, 2, 0)
            pred = pred.flatten(2)
            location = locations[i].view(h, w, 2).unsqueeze(0)
            location = location[:, 1::2, 1::2].reshape(1, -1, 2)
            pred = torch.stack([
                location[:, :, 0] - pred[:, 0],
                location[:, :, 1] - pred[:, 1],
                location[:, :, 0] + pred[:, 2],
                location[:, :, 1] + pred[:, 3]], dim=1)

            connected_box = torch.matmul(pred, pred_relations[i].permute(0, 2, 1))
            # connected_box = F.interpolate(connected_box.view(n, c, h//2, w//2), size=(h, w), mode="nearest").flatten(2)
            # connected_box = connected_box.flatten(2)
            location = locations[i].unsqueeze(0)
            connected_box = torch.stack([
                location[:, :, 0] - connected_box[:, 0],
                location[:, :, 1] - connected_box[:, 1],
                connected_box[:, 2] - location[:, :, 0],
                connected_box[:, 3] - location[:, :, 1]
            ])
            connected_box = torch.max(
                init_pred.view(n, c, h, w),
                connected_box.view(n, c, h, w))

            connected_boxes.append(permute_to_N_HW_K(connected_box, 4))
        connected_boxes = cat(connected_boxes, dim=1).view(-1, 4)

        pred_scores, pred_boxes, pred_confidences, strides =\
            permute_and_concat(pred_scores, pred_boxes, pred_confidences, strides)
        pred_confidences = 1 - (pred_confidences - pred_confidences)

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
            loss_relation = connected_boxes[foreground_idxs].sum()
        else:
            loss_box = smooth_l1_loss(
                pred_boxes[foreground_idxs],
                gt_boxes[foreground_idxs],
                beta=self.smooth_l1_loss_beta,
                reduction="none") / strides[foreground_idxs]
            loss_box, loss_box_indices = loss_box.sort(dim=1)

            loss_relation = smooth_l1_loss(
                connected_boxes[foreground_idxs],
                gt_boxes[foreground_idxs],
                beta=self.smooth_l1_loss_beta,
                reduction="none") / strides[foreground_idxs]

            # loss_relations_w = (self.loss_nominator / torch.clamp(loss_box.detach().min(dim=1)[0], min=1e-6))
            # loss_relations_w = torch.sigmoid(loss_relations_w).unsqueeze(1)
            # get_event_storage().put_scalar("relation_weights", loss_relations_w.mean())
            # loss_relation = loss_relations_w * loss_relation

            loss_box = loss_box[:, :-1].sum() +\
                pred_boxes[foreground_idxs].gather(1, loss_box_indices[:, -1:]).sum() * 1e-3
            loss_box = loss_box / num_pos_avg_per_gpu

            loss_relation = smooth_l1_loss(
                connected_boxes[foreground_idxs],
                gt_boxes[foreground_idxs],
                beta=self.smooth_l1_loss_beta,
                reduction="none") / strides[foreground_idxs]
            loss_relation = loss_relation.sum() / num_pos_avg_per_gpu * 0.5

        return {"loss_cls": loss_cls,
                "loss_box": loss_box,
                "loss_relation": loss_relation,
                "loss_confidence": pred_confidences.mean()}

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
        self.confidences = nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        self.add_module("cls_head", nn.Sequential(*cls_head))
        self.add_module("box_head", nn.Sequential(*box_head))

        # initialization
        for modules in [self.cls_head, self.box_head,
                        self.logits, self.boxes, self.confidences]:
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
        confidences = []

        for l, feature in enumerate(x):
            logits.append(self.logits(self.cls_head(feature)))
            box_feature = self.box_head(feature)
            box_pred = self.scales[l](self.boxes(box_feature))
            boxes.append(torch.exp(self.scales[l](box_pred)))
            confidences.append(self.confidences(box_feature))
        return logits, boxes, confidences


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
            masks_per_feature_level = torch.ones((N, int(np.floor(H/2)), int(np.floor(W/2))), dtype=torch.bool, device=device)
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
        parts = F.max_pool2d(parts, 2, stride=2, padding=0)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        num_pos_feats = self.num_pos_feats // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=parts.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        num_pos_feats = num_pos_feats // 2
        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=parts.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / num_pos_feats)
        pos_d = parts[:, :, :, :, None] / dim_t
        pos_d = pos_d.permute((0, 2, 3, 1, 4))
        pos_d = torch.stack((pos_d[:, :, :, :, 0::2].sin(), pos_d[:, :, :, :, 1::2].cos()), dim=5).flatten(3)
        pos = torch.cat((pos_y, pos_x, pos_d), dim=3).permute(0, 3, 1, 2)

        return pos
