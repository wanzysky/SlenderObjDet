from detectron2.modeling.backbone import BACKBONE_REGISTRY
from detectron2.modeling.backbone import build_resnet_backbone
from detectron2.modeling.backbone.fpn import FPN, LastLevelP6P7
from detectron2.layers import ShapeSpec


@BACKBONE_REGISTRY.register()
def build_retinanet_resnet_fpn_backbone_use_p5(cfg, input_shape: ShapeSpec):
    """
    Args:
        cfg: a detectron2 CfgNode

    Returns:
        backbone (Backbone): backbone module, must be a subclass of :class:`Backbone`.
    """
    bottom_up = build_resnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    in_channels_p6p7 = out_channels
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelP6P7(in_channels_p6p7, out_channels, in_feature="p5"),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone
