from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling.meta_arch import META_ARCH_REGISTRY
from detectron2.structures import ImageList, Instances, Boxes
from detectron2.layers import ShapeSpec, batched_nms, cat
from detectron2.modeling.postprocessing import detector_postprocess

from ..backbone import Backbone, build_backbone


@META_ARCH_REGISTRY.register()
class CornerNet(nn.Module):

    @configurable
    def __init__(self, *, backbone: Backbone):
        super().__init__()
        self.backbone = backbone

        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

    def from_config(self, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, *input):
        raise NotImplementedError

    def losses(self):
        raise NotImplementedError

    @torch.no_grad()
    def get_ground_truth(self):
        raise NotImplementedError

    def inference(self):
        raise NotImplementedError

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images


class CornerNetTail(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()

    def forward(self, *input):
        raise NotImplementedError
