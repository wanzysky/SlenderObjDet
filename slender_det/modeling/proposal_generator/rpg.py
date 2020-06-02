"""
Implementation of RepPoints as a RPN: (https://arxiv.org/pdf/1904.11490).

The points are now used as region proposals only.
This file incluse classes:
    RepPointsGenerator: Counterpart of RPN. It takes features from multilevel in, and produces losses and proposals.
    RepPointsInitHead: The initial prediction head of RepPoints. It predicts the objectness logits and offsets at each point. The offsets are transformed to bboxes for presentation and loss computation.
    RepPointsGeneratorResult: Loss computation and prediction assembling.
"""
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.modeling import PROPOSAL_GENERATOR_REGISTRY
from detectron2.layers import ShapeSpec
from detectron2.structures import Instances
from detectron2.modeling.proposal_generator.rpn_outputs import find_top_rpn_proposals
from detectron2.utils.registry import Registry
from detectron2.utils.events import get_event_storage

from slender_det.modeling.grid_generator import zero_center_grid, uniform_grid

REP_POINTS_HEAD_REGISTRY = Registry("RepPointsHead")
REP_POINTS_HEAD_REGISTRY.__doc__ = """
Registry for RepPoints heads, which take feature maps and perform
objectness classification and bounding box regression for each point.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


def build_rep_points_head(cfg, input_shape):
    """
    Build a RepPoints head defined by `cfg.MODEL.REP_POINTS_GENERATOR.HEAD_NAME`.
    """
    name = cfg.MODEL.PROPOSAL_GENERATOR.HEAD_NAME
    return REP_POINTS_HEAD_REGISTRY.get(name)(cfg, input_shape)


@REP_POINTS_HEAD_REGISTRY.register()
class RepPointsInitHead(nn.Module):
    kernel_size = 3

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        # Share head among levels
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        self.in_channels = in_channels[0]

        # 1, 18, 1, 1 for now, where 18 should be decomposed to h=3, w=3, c=2(x, y)
        self.register_buffer(
            "grid_cell",
            zero_center_grid(self.kernel_size).view(1, -1, 1, 1))
        number_cell_points = self.grid_cell.size(1)

        self.objectness_logits = nn.Sequential(
            *self.stacked_convs(),
            nn.Conv2d(
                self.in_channels,
                1,
                kernel_size=3,
                stride=1,
                padding=1))
        self.deltas = nn.Sequential(
            *self.stacked_convs(),
            nn.Conv2d(
                self.in_channels,
                number_cell_points,
                kernel_size=1,
                stride=1))

        self.moment_transfer = nn.Parameter(
            data=torch.zeros(2), requires_grad=True)
        self.moment_mul = 0.01

    def stacked_convs(self, layers=3):
        convs = []
        for _ in range(layers):
            convs.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, self.in_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(self.in_channels),
                    nn.ReLU(inplace=True)
                ))
        return convs

    def forward(self, features):
        """
        Args:
            features (list[Tensor]): list of feature maps
        """
        kernel_offsets = self.grid_cell.type_as(features[0])
        objectness_logits = []
        deltas = []
        for x in features:
            objectness_logits.append(self.objectness_logits(x))
            deltas.append(self.deltas(x))
        return objectness_logits, deltas

    def points2bbox(self, base_grids: torch.Tensor, deltas: List[torch.Tensor]):
        bboxes = []
        H_i, W_i = float("inf"), float("inf")
        start = 0
        # For each level
        for delta in deltas:
            """
            delta: (N, 18, H_i, W_i), 
            """
            # Assuming that strides go from small to large
            assert delta.size(-2) < H_i
            assert delta.size(-1) < W_i
            H_i, W_i = delta.shape[-2:]
            # (1, 2, H_i, W_i), grid for this feature level.
            base_grid = base_grids[start:start + H_i * W_i].view(1, H_i, W_i, 2).permute(0, 3, 1, 2)
            start += H_i * W_i

            # (N*9, 2, H_i, W_i)
            delta = delta.view(-1, 9, 2, H_i, W_i).view(-1, 2, H_i, W_i)
            # (N, 9, 2, H_i, W_i)
            points = (delta + base_grid).view(-1, 9, 2, H_i, W_i)
            # (N, 1, 2, H_i, W_i)
            points_mean = points.mean(dim=1, keepdim=True)
            # (N, 1, 2, H_i, W_i)
            points_std = torch.std(points - points_mean, dim=1, keepdim=True)
            # (2)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                    self.moment_transfer.detach() * (1 - self.moment_mul))
            half_shape = points_std * torch.exp(moment_transfer)[None, None, :, None, None]
            # (N, 4, H_i, W_i)
            bbox = torch.cat(
                [points_mean[:, :, 0] - half_shape[:, :, 0],
                 points_mean[:, :, 1] - half_shape[:, :, 1],
                 points_mean[:, :, 0] + half_shape[:, :, 0],
                 points_mean[:, :, 1] + half_shape[:, :, 1]],
                dim=1)
            bboxes.append(bbox)
        return bboxes


class RepPointsGeneratorResult():
    def __init__(
            self,
            pred_objectness_logits,
            pred_bboxes,
            gt_labels=None,
            gt_boxes=None):
        """
        Args:
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, 1, Hi, Wi) representing
                the predicted objectness logits for anchors.
            pred_bboxes (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, 4, Hi, Wi) representing the predicted bboxes.
            gt_labels (Tensor): (N, X). An array of objectness label for all positions from L levels.
            gt_boxes (Tensor): (N, X, 4). Mathced bbox for all positions.
        """
        self.pred_objectness_logits = pred_objectness_logits
        self.pred_bboxes = pred_bboxes

        self.gt_labels = gt_labels
        self.gt_boxes = gt_boxes

    def losses(self, strides):
        # (N, X)
        pred_objectness_logits = torch.cat(
            [p.view(p.size(0), -1) for p in self.pred_objectness_logits], dim=1)
        # (N, X, 4)
        pred_bboxes = torch.cat(
            [p.view(p.size(0), 4, -1) for p in self.pred_bboxes], dim=2)

        pos_masks = self.gt_labels > 0
        pos_count = pos_masks.sum()
        neg_masks = ~pos_masks
        neg_count = torch.min(neg_masks.sum(), pos_count * 3).item()

        cls_loss = sigmoid_focal_loss_jit(
            pred_objectness_logits,
            self.gt_labels,
            alpha=0.25,
            reduction="none")
        neg_cls_loss, _ = cls_loss[neg_masks].topk(neg_count)
        cls_loss = cls_loss[pos_masks].mean() + neg_cls_loss.mean()
        # (N, X)
        pred_bboxes = pred_bboxes.permute(0, 2, 1) / strides[None, :, None] / 4
        gt_bboxes = self.gt_boxes / strides[None, :, None] / 4
        localization_loss = smooth_l1_loss(
            pred_bboxes[pos_masks],
            gt_bboxes[pos_masks],
            0.11,
            reduction="mean")
        return {"cls_loss": cls_loss, "localization_loss": localization_loss}

    def predict_proposals(self):
        return [p.view(p.size(0), 4, -1).permute(0, 2, 1) for p in self.pred_bboxes]

    def predict_objectness_logits(self):
        return [p.view(p.size(0), -1) for p in self.pred_objectness_logits]


@PROPOSAL_GENERATOR_REGISTRY.register()
class RepPointsGenerator(nn.Module):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
        Inspired by the official implementation: (https://github.com/microsoft/RepPoints).
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
        self.in_features = cfg.MODEL.RPN.IN_FEATURES
        self.nms_thresh = cfg.MODEL.RPN.NMS_THRESH
        self.batch_size_per_image = cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE
        self.positive_fraction = cfg.MODEL.RPN.POSITIVE_FRACTION
        self.smooth_l1_beta = cfg.MODEL.RPN.SMOOTH_L1_BETA
        self.loss_weight = cfg.MODEL.RPN.LOSS_WEIGHT
        # fmt: on

        # Map from self.training state to train/test settings
        self.pre_nms_topk = {
            True: cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.PRE_NMS_TOPK_TEST,
        }
        self.post_nms_topk = {
            True: cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN,
            False: cfg.MODEL.RPN.POST_NMS_TOPK_TEST,
        }
        self.boundary_threshold = cfg.MODEL.RPN.BOUNDARY_THRESH
        if cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'points':
            from .matcher import nearest_point_match
            self.matcher = nearest_point_match
        else:
            assert cfg.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE == 'inside'
            from .matcher import inside_match
            self.matcher = inside_match

        self.init_head = build_rep_points_head(cfg, [input_shape[f] for f in self.in_features])

        self.strides = [input_shape[f].stride for f in self.in_features]
        grid = uniform_grid(2048)
        self.register_buffer("grid", grid)

        self.debug = cfg.DEBUG

    def get_center_grid(self, features):
        point_centers = []
        strides = []
        for f_i, feature in enumerate(features):
            height, width = feature.shape[2:]
            stride = self.strides[f_i]
            # HxW, 2
            grid = self.grid[:height, :width].reshape(-1, 2)
            strides.append(torch.full((grid.shape[0],), stride, device=grid.device))
            point_centers.append(grid * stride)
        return torch.cat(point_centers, dim=0), torch.cat(strides, dim=0)

    @torch.no_grad()
    def label_and_sample_points(
            self,
            centers: torch.Tensor,
            gt_instances: List[Instances],
            strides: torch.Tensor):
        """
        Args:
            centers (torch.Tensor): Shape (X, 2), point_centers for all positions at each feature map.
            strides (torch.Tensor): Shape (X), corresponding strides for all positions at each feature map.
            gt_instances: the ground-truth instances for each image.
        Returns:
            Tensor:
                (N, X), where X is the number of all positions from feature levels.
                It represents the objectness target at positions, 0 for negative and 1 for positive.
            Tensor:
                (N, X, 4). It represents the bbox target at each position.

        NOTE: This function follows the official implementation
            "https://github.com/microsoft/RepPoints/blob/master/src/reppoints_assigner/point_assigner.py".
            However, it remains sub-optimal in my view.
            (i.) The naive distance measure between center points of ground-truth and grid may lead to sub-sampling of small objects.
            (ii.) Each bbox of an object assign only 1 point as positive while others are marked as background.
        """
        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        objecness_labels = []
        bbox_labels = []

        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            Loop over batch size for saving memory.
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            objectness_label, bbox_label = self.matcher(centers, strides, gt_boxes_i)
            objecness_labels.append(objectness_label)
            bbox_labels.append(bbox_label)
        return torch.stack(objecness_labels, dim=0), torch.stack(bbox_labels, dim=0)

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

        pred_objectness_logits, pred_deltas = self.init_head(features)
        torch.cuda.synchronize()

        point_centers, strides = self.get_center_grid(features)
        point_centers = point_centers.to(pred_deltas[0].device)
        strides = strides.to(pred_deltas[0].device)
        pred_boxes = self.init_head.points2bbox(point_centers, pred_deltas)
        if self.training:
            # (N, H*W*L), (N, H*W*L, 4)
            gt_labels, gt_boxes = self.label_and_sample_points(
                point_centers, gt_instances, strides)
        else:
            gt_labels, gt_boxes = None, None

        outputs = RepPointsGeneratorResult(
            pred_objectness_logits,
            pred_boxes,
            gt_labels,
            gt_boxes)

        if self.training:
            losses = {k: v * self.loss_weight for k, v in outputs.losses(strides).items()}
        else:
            losses = {}

        proposals = outputs.predict_proposals()
        logits = outputs.predict_objectness_logits()
        if self.debug:
            storage = get_event_storage()
            start = 0
            for i, f in enumerate(features):
                h, w = f.shape[-2:]
                centers = point_centers[start:start + h * w].view(h, w, 2)
                stride = strides[start:start + h * w].view(h, w)
                storage.put_image("centers_x-%d" % i, (centers[..., 0:1] / centers[..., 0:1].max()).permute(2, 0, 1))
                storage.put_image("centers_y-%d" % i, (centers[..., 1:] / centers[..., 1:].max()).permute(2, 0, 1))
                storage.put_image("strides-%d" % i, (stride[None] / 64).float())

                gt_label = gt_labels[0, start:start + h * w].view(1, h, w)
                storage.put_image("gt-labels-%d" % i, gt_label.float())

                storage.put_image("pred-logits-%d" % i, torch.sigmoid(logits[i][0].view(1, h, w)))

                start += h * w
            # storage.clear_images()

        with torch.no_grad():
            # Find the top proposals by applying NMS and removing boxes that
            # are too small. The proposals are treated as fixed for approximate
            # joint training with roi heads. This approach ignores the derivative
            # w.r.t. the proposal boxes’ coordinates that are also network
            # responses, so is approximate.
            proposals = find_top_rpn_proposals(
                proposals,
                logits,
                images,
                self.nms_thresh,
                self.pre_nms_topk[self.training],
                self.post_nms_topk[self.training],
                self.min_box_side_len,
                self.training,
            )

        return proposals, losses
