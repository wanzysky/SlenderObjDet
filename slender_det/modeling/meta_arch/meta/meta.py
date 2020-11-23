import logging
import math
import copy
import numpy as np
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fvcore.nn import sigmoid_focal_loss_jit, smooth_l1_loss

from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.config import configurable
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.layers import DeformConv
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.logger import log_first_n

from .heads import build_meta_head


@META_ARCH_REGISTRY.register()
class AblationMetaArch(nn.Module):
    """
    Meta Architecture for Ablation Study.
        Box representation:
            - Anchor + offsets
            - Anchor Free
            - Points set
        Feature Adaption:
            - None(Baseline)
            - Unsupervised DCN
            - Supervised DCN
    """

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            head: nn.Module,
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
            in_features
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head

        self.input_format = input_format
        self.vis_period = vis_period
        self.in_features = in_features
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1))
        assert (
                self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        params = cfg.MODEL.META_ARCH

        backbone_shape = backbone.output_shape()
        feature_shapes = [backbone_shape[f] for f in params.IN_FEATURES]
        head = build_meta_head(cfg, feature_shapes)
        in_features = cfg.MODEL.RETINANET.IN_FEATURES

        return {
            "backbone": backbone,
            "head": head,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "in_features": in_features
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of : class:`DatasetMapper`.
            Each item in the list contains the input for one image.
            For now, each item in the list is a dict that contains:
             * images: Tensor, image in (C, H, W) format.
             * instances: Instances.
             Other information that' s included in the original dict ,such as:
             * "height", "width"(int): the output resolution of the model,
             used in inference.See  `postprocess` for detail
        Return:
            dict[str: Tensor]:
                mapping from a named loss to a tensor storing the loss, Used
                during training only.
                At inference stage, return predicted bboxes.
        """
        if not self.training:
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
        elif "targets" in batched_inputs[0]:
            log_first_n(logging.WARN, "'targets' in the model inputs is now renamed to 'instances'!", n=10)
            gt_instances = [
                x['instances'].to(self.device) for x in batched_inputs
            ]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]
        results = self.head(images, features, gt_instances=gt_instances)
        if self.training:
            return results
        
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def inference(self, batched_inputs):
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = [features[f] for f in self.head.in_features]
        results = self.head(images, features)
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
                results, batched_inputs, images.image_sizes):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x['image'].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        return images
