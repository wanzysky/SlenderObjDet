from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.modeling.proposal_generator.rpn_outputs import find_top_rpn_proposals


def likelyhood_loss(target: torch.Tensor, coordinates: torch.Tensor, mask=None):
    """
    Compute loss based on maximum likelyhood esimitation.
    Args:
        target: (Tensor): Target distribution with maximum value 1, whose shape is (N, H, W).
        indices: (Tensor): Predicted sampling points with shape (N, P, 2, H, W) on the target distribution.
    """
    # (N, PxH, W, 2)
    coordinates = coordinates.permute(0, 1, 3, 4, 2)
    N, P, H, W = coordinates.shape[:-1]
    coordinates = coordinates.view(N, P * H, W, 2)
    likelyhood = F.grid_sample(target, coordinates).view(N, P, H, W)
    if mask:
        likelyhood = likelyhood * mask
    return -torch.log(likelyhood.mean(2).mean(2) + 1e-8)


def offsets2coordinates(offsets: torch.Tensor, image_shape):
    """
    Args:
        offsets (Tensor): Shape (N, 18, H, W)
    Returns:
        indices (Tensor:Long): Shape (N, 9, 2, H, W)
    """
    ys, xs = torch.meshgrid(
        torch.linspace(0, image_shape[1] - 1, offsets.shape[-2]),
        torch.linspace(0, image_shape[0] - 1, offsets.shape[-1]))
    offsets = offsets.view(-1, 9, 2, offsets.shape[-2], offsets.shape[-1])
    coordinates = torch.stack([xs, ys], dim=0).view(1, 1, 2, xs.shape[-2], xs.shape[-1])
    return offsets + coordinates


class PointsProposalOutputs(object):
    def __init__(
        self,
        images,
        pred_logists,
        pred_offsets,
        gt_centers=None,
        gt_borders=None,
        gt_logits=None
    ):
        self.image_sizes = images.image_sizes
        self.pred_logists = pred_logists
        self.pred_offsets = pred_offsets
        self.pred_coordinates = offsets2coordinates(pred_offsets, images.tensor.shape[-2:])

        self.gt_centers = gt_centers
        self.gt_borders = gt_borders
        self.gt_logits = gt_logits


    def losses(self, with_center_l1=False):
        object_logits_loss = F.binary_cross_entropy_with_logits(
            self.pred_logists,
            self.gt_logits,
            reduction="sum")
        border_points = torch.cat(
            [self.pred_coordinates[:4], self.pred_coordinates[5:]],
            dim=1)
        center_points = self.pred_coordinates[4:5]
        border_likely_loss = likelyhood_loss(self.gt_borders, border_points)
        center_likely_loss = likelyhood_loss(self.gt_centers, center_points)
        return {"object_loss": object_logits_loss, "border_loss": border_likely_loss, "center_loss": center_likely_loss}

    def predict_proposals(self):
        N = self.pred_coordinates.shape[0]
        # (N, H*W)
        xmin = self.pred_coordinates[:, :, 0].min(1).view(N, -1)
        ymin = self.pred_coordinates[:, :, 1].min(1).view(N, -1)
        xmax = self.pred_coordinates[:, :, 0].max(1).view(N, -1)
        ymax = self.pred_coordinates[:, :, 1].max(1).view(N, -1)
        return torch.stack([xmin, ymin, xmax, ymax], dim=-1)

    def predict_objectness_logits(self):
        return self.pred_logists.view(self.pred_logists.shape[0], -1)


@PROPOSAL_GENERATOR_REGISTRY.register()
class PointsProposalGenerator(nn.Module):
    """
    Points proposal generator, and changes are ongoing.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()
        self.in_features = cfg.MODEL.ProposalGenerator.IN_FEATURES
        self.num_points = cfg.MODEL.ProposalGenerator.NUM_POINTS

        # We follow RPN in sharing across levels
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting offsets, (x, y) for each point.
        self.offsets = nn.Conv2d(in_channels, self.num_points * 2, kernel_size=1, stride=1)
        # 1x1 conv for predicting if-inside-object logits
        self.in_object_logits = nn.Conv2d(in_channels, 1, kernel_size=1, stride=1)

    def rescale(self, offsets, feature, images):
        scale_x = images.tensor.shape[-1] / feature.shape[-1]
        scale_y = images.tensor.shape[-2] / feature.shape[-2]
        assert scale_y == scale_x
        scale = scale_x

        return (torch.exp(offsets) - 1) * scale * self.base_offset

    def forward(self, images, features, gt_instances=None):
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

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses().items()}
        else:
            losses = {}

        outputs = PointsProposalOutputs(
            images,
            pred_in_object_logits,
            pred_offsets,
            gt_instances.centers,
            gt_instances.borders,
            gt_instances.logits)

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

        return proposals, losses
