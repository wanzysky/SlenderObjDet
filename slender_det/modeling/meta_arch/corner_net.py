from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec, batched_nms, cat, Conv2d, get_norm
from detectron2.modeling.postprocessing import detector_postprocess
from slender_det.layers import TLPool, BRPool

from ..backbone import Backbone, build_backbone


@META_ARCH_REGISTRY.register()
class CornerNet(nn.Module):

    @configurable
    def __init__(
            self, *,
            in_features: Tuple[str],
            backbone: Backbone,
            head: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
    ):
        super().__init__()
        self.in_features = in_features
        self.backbone = backbone
        self.head = head

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))

    @classmethod
    def from_config(cls, cfg):
        in_features = cfg.MODEL.CORNER_NET.IN_FEATURES

        backbone = build_backbone(cfg)
        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in in_features]
        head = CornerNetHead(cfg, feature_shapes)

        return {
            "backbone": backbone,
            "head": head,
            "in_features": in_features,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs = self.head(features)

        if self.training:
            assert "instances" in batched_inputs[0], "Instance annotations are missing in training!"
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]

            self.get_ground_truth(gt_instances)

    def losses(self, predictions, ground_truths):
        raise NotImplementedError

    @torch.no_grad()
    def get_ground_truth(self, gt_instances):
        import pdb
        pdb.set_trace()

    def inference(self, tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs):
        raise NotImplementedError

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class CornerNetHead(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

        in_channels = input_shape[0].channels
        num_classes = cfg.MODEL.CORNER_NET.NUM_CLASSES
        stacks = len(input_shape)
        norm = cfg.MODEL.CORNER_NET.NORM

        self.tl_convs = nn.ModuleList([TLPool(in_channels) for _ in range(stacks)])
        self.br_convs = nn.ModuleList([BRPool(in_channels) for _ in range(stacks)])

        self.tl_heats = nn.ModuleList([self.make_stage(in_channels, num_classes, norm) for _ in range(stacks)])
        self.br_heats = nn.ModuleList([self.make_stage(in_channels, num_classes, norm) for _ in range(stacks)])

        self.tl_tags = nn.ModuleList([self.make_stage(in_channels, 1, norm) for _ in range(stacks)])
        self.br_tags = nn.ModuleList([self.make_stage(in_channels, 1, norm) for _ in range(stacks)])

        self.tl_offsets = nn.ModuleList([self.make_stage(in_channels, 2, norm) for _ in range(stacks)])
        self.br_offsets = nn.ModuleList([self.make_stage(in_channels, 2, norm) for _ in range(stacks)])

    def make_stage(self, in_channels, out_channels, norm):
        layers = [
            Conv2d(
                in_channels, in_channels, 3,
                padding=1, bias=False,
                norm=get_norm(norm, in_channels),
                activation=F.relu_
            ),
            Conv2d(in_channels, out_channels, 1)
        ]

        return nn.Sequential(*layers)

    def forward(self, features: List[torch.Tensor]):
        tl_feats = [tl_conv(feature) for tl_conv, feature in zip(self.tl_convs, features)]
        br_feats = [br_conv(feature) for br_conv, feature in zip(self.br_convs, features)]

        tl_heats = [tl_heat(tl_feat) for tl_heat, tl_feat in zip(self.tl_heats, tl_feats)]
        br_heats = [br_heat(br_feat) for br_heat, br_feat in zip(self.br_heats, br_feats)]

        tl_tags = [tl_tag_(tl_feat) for tl_tag_, tl_feat in zip(self.tl_tags, tl_feats)]
        br_tags = [br_tag_(br_feat) for br_tag_, br_feat in zip(self.br_tags, br_feats)]

        tl_offs = [tl_off_(tl_feat) for tl_off_, tl_feat in zip(self.tl_offsets, tl_feats)]
        br_offs = [br_off_(br_feat) for br_off_, br_feat in zip(self.br_offsets, br_feats)]

        return [tl_heats, br_heats, tl_tags, br_tags, tl_offs, br_offs]

