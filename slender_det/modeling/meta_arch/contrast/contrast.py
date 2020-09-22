from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.structures import ImageList
from detectron2.layers import ShapeSpec

from slender_det.modeling.backbone import build_backbone, Backbone


@META_ARCH_REGISTRY.register()
class ContrastRPN(nn.Module):

    @configurable
    def __init__(
            self,
            *,
            backbone: Backbone,
            head: nn.Module,
            in_features: Tuple[str],
            fpn_stride: Tuple[int],
            pixel_mean: Tuple[float],
            pixel_std: Tuple[float],
            input_format: Optional[str] = None,
            vis_period: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = head
        self.in_features = in_features
        self.fpn_stride = fpn_stride

        self.input_format = input_format
        self.vis_period = vis_period
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
        output_shape = backbone.output_shape()

        head = ContrastRPNHead(cfg, output_shape)
        model_params = cfg.MODEL.CONTRAST
        in_features = model_params.IN_FEATURES
        fpn_stride = [output_shape[in_feature].stride for in_feature in in_features]
        return {
            "backbone": backbone,
            "head": head,
            "in_features": in_features,
            "fpn_stride": fpn_stride,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = [features[f] for f in self.in_features]

        cls_outs, embed_outs = self.head(features)

        import ipdb; ipdb.set_trace()
        
    def losses(self, targets):
        return None

    @torch.no_grad()
    def get_ground_truth(self, targets):
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


class ContrastRPNHead(nn.Module):

    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        super().__init__()
        self.in_channels = input_shape[0].channels

        model_params = cfg.MODEL.CONTRAST
        self.stacked_convs = model_params.STACK_CONVS
        self.feat_channels = model_params.FEAT_CHANNELS
        self.norm_mode = model_params.NORM_MODE
        self.embed_channels = model_params.EMBED_CHANNELS

        self.cls_branch = nn.ModuleList()
        self.ebd_branch = nn.ModuleList()

        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_branch.append(
                nn.Conv2d(chn,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            if self.norm_mode == 'GN':
                self.cls_branch.append(
                    nn.GroupNorm(32 * self.feat_channels // 256, self.feat_channels))
            else:
                raise ValueError('The normalization method in reppoints head should be GN')
            self.cls_branch.append(nn.ReLU(inplace=True))

            self.ebd_branch.append(
                nn.Conv2d(chn,
                          self.feat_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=False))
            if self.norm_mode == 'GN':
                self.ebd_branch.append(
                    nn.GroupNorm(32 * self.feat_channels // 256, self.feat_channels))
            else:
                raise ValueError('The normalization method in reppoints head should be GN')
            self.ebd_branch.append(nn.ReLU(inplace=True))

        self.cls_conv = nn.Conv2d(self.feat_channels, self.feat_channels, 3, 1, 1)
        self.cls_out = nn.Conv2d(self.feat_channels, 2, 1, 1, 0)
        self.emd_out = nn.Conv2d(self.feat_channels, self.embed_channels, 1, 1, 0)
        self.init_weights()

    def init_weights(self):
        """
        Initialize model weights
        """
        for modules in [
            self.cls_branch, self.ebd_branch, self.cls_conv, self.emd_out
        ]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

    def forward(self, features):
        cls_outs = []
        ebd_outs = []

        for l, feature in enumerate(features):
            cls_feat = feature
            ebd_feat = feature

            for cls_conv in self.cls_branch:
                cls_feat = cls_conv(cls_feat)
            for ebd_conv in self.ebd_branch:
                ebd_feat = ebd_conv(ebd_feat)

            cls_out = self.cls_out(F.relu(self.cls_conv(cls_feat)))
            cls_outs.append(cls_out)

            # TODO: add normalization for embedded feature?
            ebd_out = self.emd_out(ebd_feat)
            ebd_outs.append(ebd_out)

        return cls_outs, ebd_outs