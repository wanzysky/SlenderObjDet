from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

# -------------------------------------------------------------------------------#
# Required by SlenderDet
# -------------------------------------------------------------------------------#

_C.DEBUG = False

_C.USE_NORI = False
_C.NEED_MASKS = False
_C.NORI_PATH = "s3://detection/datasets/coco/"
_C.REDIS = CN()
_C.REDIS.HOST = "127.0.0.1"
_C.REDIS.PORT = 6379
_C.REDIS.DB = 0

_C.MODEL.PROPOSAL_GENERATOR.IN_FEATURES = ["p5"]
_C.MODEL.PROPOSAL_GENERATOR.NUM_POINTS = 9
_C.MODEL.PROPOSAL_GENERATOR.SIZES = [8, 16, 32, 64, 128]
_C.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE = "point"

_C.MODEL.PROPOSAL_GENERATOR.HEAD_NAME = ""

# ---------------------------------------------------------------------------- #
# FCOS Options
# ---------------------------------------------------------------------------- #
_C.MODEL.FCOS = CN()
# the number of classes excluding background
_C.MODEL.FCOS.NUM_CLASSES = 80
_C.MODEL.FCOS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_C.MODEL.FCOS.FPN_STRIDES = [8, 16, 32, 64, 128]
_C.MODEL.FCOS.PRIOR_PROB = 0.01
_C.MODEL.FCOS.INFERENCE_TH = 0.05
_C.MODEL.FCOS.NMS_TH = 0.6
_C.MODEL.FCOS.PRE_NMS_TOP_N = 1000

# Focal loss parameter: alpha
_C.MODEL.FCOS.FOCAL_LOSS_ALPHA = 0.25
# Focal loss parameter: gamma
_C.MODEL.FCOS.FOCAL_LOSS_GAMMA = 2.0

# the number of convolutions used in the cls and bbox tower
_C.MODEL.FCOS.NUM_CONVS = 4

# if CENTER_SAMPLING_RADIUS <= 0, it will disable center sampling
_C.MODEL.FCOS.CENTER_SAMPLING_RADIUS = 0.0
# IOU_LOSS_TYPE can be "iou", "linear_iou" or "giou"
_C.MODEL.FCOS.IOU_LOSS_TYPE = "iou"

_C.MODEL.FCOS.NORM_REG_TARGETS = False
_C.MODEL.FCOS.CENTERNESS_ON_REG = False

_C.MODEL.FCOS.USE_DCN_IN_TOWER = False
_C.MODEL.FCOS.USE_DCN_V2 = True

# FCOS Anchor Setting
_C.MODEL.FCOS.SMOOTH_L1_LOSS_BETA = 0.1


def get_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C
