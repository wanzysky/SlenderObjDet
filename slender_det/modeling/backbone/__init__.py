from detectron2.modeling.backbone import Backbone, build_backbone

from .fpn import (
    build_retinanet_pvt_fpn_backbone,
    build_retinanet_resnet_fpn_backbone_use_p5,
    build_retinanet_resnet_vt_fpn,
    build_retinanet_resnet_vt_fpn_backbone_use_p5,
)
from .hourglass import build_hourglass_backbone
from .pvt import build_pvt_backbone
