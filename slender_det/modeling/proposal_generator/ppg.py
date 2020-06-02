from typing import Dict
from collections import defaultdict

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals
from detectron2.utils.events import get_event_storage


def likelyhood_loss(target: torch.Tensor, coordinates: torch.Tensor, mask=None):
    """
    Compute loss based on maximum likelyhood esimitation.
    Args:
        target: (Tensor): Target distribution with maximum value 1, whose shape is (N, H, W).
        indices: (Tensor): Predicted sampling points with shape (N, P, 2, H, W) on the target distribution.
    """
    target = target.unsqueeze(1)
    # (N, PxH, W, 2)
    coordinates = coordinates.permute(0, 1, 3, 4, 2)
    N, P, H, W = coordinates.shape[:-1]
    coordinates = coordinates.reshape(N, P * H, W, 2)
    likelyhood = F.grid_sample(target, coordinates).reshape(N, P, H, W)
    if mask is not None:
        likelyhood = likelyhood * \
                     F.grid_sample(mask.unsqueeze(1).type(torch.float32), coordinates).reshape(N, P, H, W)
    return -torch.log(likelyhood.mean(2).mean(2) + 1e-8)


def offsets2coordinates(offsets: torch.Tensor, image_shape):
    """
    Args:
        offsets (Tensor): Shape (N, 18, H, W)
    Returns:
        indices (Tensor:Long): Shape (N, 9, 2, H, W)
    """
    H, W = image_shape
    ys, xs = torch.meshgrid(
        torch.linspace(0, image_shape[1] - 1, offsets.shape[-2]),
        torch.linspace(0, image_shape[0] - 1, offsets.shape[-1]))
    xs = xs.to(offsets.device)
    ys = ys.to(offsets.device)
    offsets = offsets.view(-1, 9, 2, offsets.shape[-2], offsets.shape[-1])
    coordinates = torch.stack([xs, ys], dim=0).view(1, 1, 2, xs.shape[-2], xs.shape[-1])
    coordinates = offsets + coordinates
    return torch.stack([coordinates[:, :, 0].clamp(0, W - 1), coordinates[:, :, 1].clamp(0, H - 1)], dim=2)


class PointsProposalOutputs(object):
    def __init__(
            self,
            images,
            pred_logits,
            pred_offsets,
            gt_sizes=None,
            strides=None
    ):
        self.image_sizes = images.image_sizes
        self.pred_logits = pred_logits
        self.pred_offsets = pred_offsets
        self.pred_coordinates = [offsets2coordinates(
            offset, images.tensor.shape[-2:]) for offset in pred_offsets]

        device = self.pred_logits[0].device
        self.gt_sizes = torch.sqrt(torch.pow(gt_sizes.tensor.to(device), 2).sum(1))
        self.strides = strides

        self.num_feature_maps = len(pred_logits)

        storage = get_event_storage()
        storage.put_image("sizes", self.gt_sizes[0:1] / 512)

    def gt_logit(self, lower, upper, size, stride, use_grid=False):
        assert stride > 1
        if use_grid:
            ys, xs = torch.meshgrid(
                torch.linspace(0, size[0] - 1, size[0]),
                torch.linspace(0, size[1] - 1, size[1]))
            grid = (torch.stack([xs, ys], 2).to(self.gt_sizes.device)) * stride
            grid = grid.view(1, *size, 2).repeat(self.gt_sizes.shape[0], 1, 1, 1)

            gt_logit = F.grid_sample(self.gt_sizes[:, None], grid, mode="nearest")
        else:
            gt_logit = F.interpolate(self.gt_sizes[:, None], scale_factor=1 / stride, mode="nearest")
        base_gt_logit = gt_logit.eq(0).float() - 1
        gt_logit = (gt_logit > lower) * (gt_logit <= upper)
        gt_logit = gt_logit * 2 + base_gt_logit
        return gt_logit

    def losses(self, with_center_l1=False, sizes=[32, 64, 128, -1]):
        losses = dict()

        # Set all object areas as ignore.
        size_lower_limit = 0

        storage = get_event_storage()
        # Compute losses for each layer separately.
        for l in range(self.num_feature_maps):
            size_upper_limit = sizes[l]
            if size_upper_limit == -1:
                size_upper_limit = 102400

            # Assign regions in specific siz as 1.
            pred_logit = self.pred_logits[l]
            gt_logit = self.gt_logit(
                size_lower_limit,
                size_upper_limit,
                pred_logit.shape[2:],
                self.strides[l])
            size_lower_limit = size_upper_limit

            storage.put_image("gt_logits/%d" % l, (gt_logit[0] + 1) * 0.5)
            storage.put_image("pred_logits/%d" % l, torch.sigmoid(pred_logit[0]))

            pred_coordinates = self.pred_coordinates[l]
            border_points = torch.cat(
                [pred_coordinates[:, :4], pred_coordinates[:, 5:]],
                dim=1)
            center_points = pred_coordinates[:, 4:5]

            losses["border_likely_loss_%d" % l] = border_points.sum() * 0
            losses["center_likely_loss_%d" % l] = center_points.sum() * 0

            pred_logit = pred_logit.view(pred_logit.shape[0], -1)
            gt_logit = gt_logit.view(gt_logit.shape[0], -1)
            pos_masks = (gt_logit >= 0)
            losses["objectness_loss_%d" % l] = (F.binary_cross_entropy_with_logits(
                pred_logit,
                gt_logit,
                reduction="none") * pos_masks).sum() / (pos_masks.sum() + 1e-5)

        return losses

    def predict_proposals(self):
        proposals = []
        for coordinates in self.pred_coordinates:
            # N, 9, 2, H, W
            N = coordinates.shape[0]
            # (N, H*W)
            xmin = coordinates[:, :, 0].min(1)[0].view(N, -1)
            ymin = coordinates[:, :, 1].min(1)[0].view(N, -1)
            xmax = coordinates[:, :, 0].max(1)[0].view(N, -1)
            ymax = coordinates[:, :, 1].max(1)[0].view(N, -1)
            proposals.append(torch.stack([xmin, ymin, xmax, ymax], dim=-1))
        return proposals

    def predict_objectness_logits(self):
        return [x.reshape(x.shape[0], -1) for x in self.pred_logits]


@PROPOSAL_GENERATOR_REGISTRY.register()
class PointsProposalGenerator(nn.Module):
    """
    Points proposal generator, and changes are ongoing.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.PROPOSAL_GENERATOR.IN_FEATURES
        self.num_points = cfg.MODEL.PROPOSAL_GENERATOR.NUM_POINTS
        self.nms_thresh = 0.7
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT

        # We follow RPN in sharing across levels
        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]
        self.strides = [input_shape[f].stride for f in self.in_features]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting offsets, (x, y) for each point.
        self.offsets = nn.Conv2d(in_channels, self.num_points * 2, kernel_size=1, stride=1)
        nn.init.constant_(self.offsets.weight, 0)
        nn.init.constant_(self.offsets.bias, 0)
        # 1x1 conv for predicting if-inside-object logits
        self.in_object_logits = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def rescale(self, offsets, feature, images):
        scale_x = images.tensor.shape[-1] / feature.shape[-1]
        scale_y = images.tensor.shape[-2] / feature.shape[-2]
        assert scale_y == scale_x
        scale = scale_x

        return (torch.exp(offsets * scale) - 1)

    def forward(self, images, features, gt_instances=None, sizes=None):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str: Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """

        features = [features[f] for f in self.in_features]
        pred_in_object_logits = []
        pred_offsets = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_in_object_logits.append(self.in_object_logits(t))
            pred_offsets.append(self.rescale(self.offsets(t), t, images))

        outputs = PointsProposalOutputs(
            images,
            pred_in_object_logits,
            pred_offsets,
            sizes,
            strides=self.strides)

        if self.training:
            # losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
            losses = {k: v for k, v in outputs.losses().items()}
        else:
            losses = {}

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                outputs.predict_proposals(),
                outputs.predict_objectness_logits(),
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        storage = get_event_storage()
        storage.clear_images()
        return None, losses
