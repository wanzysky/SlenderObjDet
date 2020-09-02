from typing import List, Tuple

import torch
import torch.nn as nn

from detectron2.utils.registry import Registry
from detectron2.layers import ShapeSpec, get_norm, DeformConv

MEAT_HEADS_REGISTRY = Registry("META_HEADS")
MEAT_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
"""

FEAT_ADAPTION_METHODS = ["Empty", "Unsupervised Offset", "Supervised Offset"]


class HeadBase(nn.Module):
    def __init__(self, cfg, input_shape: List[ShapeSpec]):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super().__init__()
        head_params = cfg.MODEL.META_ARCH
        # TODO: Implement the sigmoid version first.
        # fmt: off
        self.in_channels = input_shape[0].channels
        self.in_features = head_params.IN_FEATURES
        self.fpn_strides = head_params.FPN_STRIDES

        self.num_classes = head_params.NUM_CLASSES
        self.feat_channels = head_params.FEAT_CHANNELS
        self.stacked_convs = head_params.STACK_CONVS
        self.norm = head_params.NORM

        self.feat_adaption = head_params.FEAT_ADAPTION
        self.res_refine = head_params.RES_REFINE
        self.loc_feat_channels = head_params.LOC_FEAT_CHANNELS
        self.gradient_mul = head_params.GRADIENT_MUL
        self.prior_prob = head_params.PRIOR_PROB
        
        cls_subnet = []
        loc_subnet = []
        for i in range(self.stacked_convs):
            in_channels = self.in_channels if i == 0 else self.feat_channels
            cls_subnet.append(
                nn.Conv2d(in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
            )
            if self.norm:
                cls_subnet.append(get_norm(self.norm, self.feat_channels))
            cls_subnet.append(nn.ReLU())
            loc_subnet.append(
                nn.Conv2d(in_channels, self.feat_channels, kernel_size=3, stride=1, padding=1)
            )
            if self.norm:
                loc_subnet.append(get_norm(self.norm, self.feat_channels))
            loc_subnet.append(nn.ReLU())

        self.cls_subnet = nn.Sequential(*cls_subnet)
        self.loc_subnet = nn.Sequential(*loc_subnet)

        # Initialization
        for modules in [self.cls_subnet, self.loc_subnet]:
            for layer in modules.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.normal_(layer.weight, mean=0, std=0.01)
                    nn.init.constant_(layer.bias, 0)
                if isinstance(layer, nn.GroupNorm):
                    nn.init.constant_(layer.weight, 1)
                    nn.init.constant_(layer.bias, 0)

        # loss parameters:
        self.focal_loss_gamma = head_params.FOCAL_LOSS_GAMMA
        self.focal_loss_alpha = head_params.FOCAL_LOSS_ALPHA
        self.loss_cls_weight = head_params.LOSS_CLS_WEIGHT
        self.loss_loc_init_weight = head_params.LOSS_LOC_INIT_WEIGHT
        self.loss_loc_refine_weight = head_params.LOSS_LOC_REFINE_WEIGHT

        # inference parameters:
        self.score_threshold = head_params.SCORE_THRESH_TEST
        self.topk_candidates = head_params.TOPK_CANDIDATES_TEST
        self.nms_threshold = head_params.NMS_THRESH_TEST
        self.max_detections_per_image = cfg.TEST.DETECTIONS_PER_IMAGE

    def make_feature_adaptive_layers(self):
        assert self.feat_adaption in FEAT_ADAPTION_METHODS, \
            "{} {}".format(self.feat_adaption, type(self.feat_adaption))
        in_channels = self.feat_channels
        if self.feat_adaption is None:
            cls_conv = nn.Conv2d(in_channels, self.feat_channels, 3, 1, 1)
            loc_conv_refine = nn.Conv2d(in_channels, self.loc_feat_channels, 3, 1, 1)
        elif self.feat_adaption == "Unsupervised Offset" or self.feat_adaption == "Supervised Offset":
            cls_conv = DeformConv(in_channels, self.feat_channels, 3, 1, 1)
            loc_conv_refine = DeformConv(in_channels, self.loc_feat_channels, 3, 1, 1)
        else:
            # TODO: refine error type
            raise ValueError("feature adaptive method undefined: {}".format(self.feat_adaption))

        return cls_conv, loc_conv_refine

    def forward(self, *input):
        raise NotImplementedError


def build_meta_head(cfg, input_shape: List[ShapeSpec]):
    return MEAT_HEADS_REGISTRY.get(cfg.MODEL.META_ARCH.NAME)(cfg, input_shape)
