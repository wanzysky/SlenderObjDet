from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

# -------------------------------------------------------------------------------#
# Required by SlenderDet
# -------------------------------------------------------------------------------#

_C.DEBUG = False

_C.USE_NORI = False
_C.NEED_MASKS = False

_C.DATALOADER.OSS_ROOT = "s3://detection/"

_C.NORI_PATH = "s3://detection/datasets/coco/"
_C.REDIS = CN()
_C.REDIS.HOST = "10.124.171.195"
_C.REDIS.PORT = 6379
_C.REDIS.DB = 0

# Matcer Type ["Origin", "TopK"]
_C.MODEL.RPN.MATCHER = CN()
_C.MODEL.RPN.MATCHER.TYPE = "Origin"
# default top k is 10
_C.MODEL.RPN.MATCHER.TOPK = 10

_C.MODEL.PROPOSAL_GENERATOR.IN_FEATURES = ["p5"]
_C.MODEL.PROPOSAL_GENERATOR.NUM_POINTS = 9
_C.MODEL.PROPOSAL_GENERATOR.SIZES = [8, 16, 32, 64, 128]
_C.MODEL.PROPOSAL_GENERATOR.SAMPLE_MODE = "point"

_C.MODEL.PROPOSAL_GENERATOR.HEAD_NAME = ""

# ---------------------------------------------------------------------------- #
# Hourglass Backbone Options used for CornerNet only
# ---------------------------------------------------------------------------- #
_C.MODEL.HOURGLASS = CN()

_C.MODEL.HOURGLASS.DEPTH = 50
_C.MODEL.HOURGLASS.STACKS = 2
_C.MODEL.HOURGLASS.OUT_FEATURES = ["hourglass2", "hourglass3"]
# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.HOURGLASS.NORM = "FrozenBN"

_C.MODEL.HOURGLASS.STEM_OUT_CHANNELS = 128

# hourglass block options
_C.MODEL.HOURGLASS.DEPTH_BLOCK = 5
_C.MODEL.HOURGLASS.CHANNELS_BLOCK = [256, 256, 384, 384, 384, 512]
_C.MODEL.HOURGLASS.NUM_CONV_BLOCK = [2, 2, 2, 2, 2, 4]

# ---------------------------------------------------------------------------- #
# CornerNet Options
# ---------------------------------------------------------------------------- #
_C.MODEL.CORNER_NET = CN()

_C.MODEL.CORNER_NET.IN_FEATURES = ["hourglass2", "hourglass3"]
_C.MODEL.CORNER_NET.NUM_CLASSES = 80
_C.MODEL.CORNER_NET.NORM = "FrozenBN"

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

# ---------------------------------------------------------------------------- #
# RepPoints Options
# ---------------------------------------------------------------------------- #
_C.MODEL.REPPOINTS = CN()

_C.MODEL.REPPOINTS.NUM_CLASSES = 80
_C.MODEL.REPPOINTS.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_C.MODEL.REPPOINTS.FPN_STRIDES = [8, 16, 32, 64, 128]

_C.MODEL.REPPOINTS.FEAT_CHANNELS = 256
_C.MODEL.REPPOINTS.POINT_FEAT_CHANNELS = 256
_C.MODEL.REPPOINTS.STACK_CONVS = 3
_C.MODEL.REPPOINTS.NORM_MODE = "GN"
_C.MODEL.REPPOINTS.GRADIENT_MUL = 0.1
_C.MODEL.REPPOINTS.PRIOR_PROB = 0.01

_C.MODEL.REPPOINTS.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.REPPOINTS.FOCAL_LOSS_ALPHA = 0.25

_C.MODEL.REPPOINTS.LOSS_CLS_WEIGHT = 1.0
_C.MODEL.REPPOINTS.LOSS_BBOX_INIT_WEIGHT = 0.5
_C.MODEL.REPPOINTS.LOSS_BBOX_REFINE_WEIGHT = 1.0
_C.MODEL.REPPOINTS.POINT_BASE_SCALE = 4
_C.MODEL.REPPOINTS.NUM_POINTS = 9
_C.MODEL.REPPOINTS.TRANSFORM_METHOD = "minmax"
_C.MODEL.REPPOINTS.MOMENT_MUL = 0.01

_C.MODEL.REPPOINTS.SCORE_THRESH_TEST = 0.05
_C.MODEL.REPPOINTS.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.REPPOINTS.NMS_THRESH_TEST = 0.5

# ---------------------------------------------------------------------------- #
# Meta Arch Options
# ---------------------------------------------------------------------------- #
_C.MODEL.META_ARCH = CN()
_C.MODEL.META_ARCH.NAME = "PointSetHead"
_C.MODEL.META_ARCH.NUM_CLASSES = 80
_C.MODEL.META_ARCH.IN_FEATURES = ['p3', 'p4', 'p5', 'p6', 'p7']
_C.MODEL.META_ARCH.FPN_STRIDES = [8, 16, 32, 64, 128]

# base convs
_C.MODEL.META_ARCH.STACK_CONVS = 3
_C.MODEL.META_ARCH.FEAT_CHANNELS = 256
_C.MODEL.META_ARCH.NORM = "GN"

# "Empty", "Unsupervised Offset", "Supervised Offset"
_C.MODEL.META_ARCH.FEAT_ADAPTION = "Empty"
_C.MODEL.META_ARCH.RES_REFINE = False

_C.MODEL.META_ARCH.LOC_FEAT_CHANNELS = 256

_C.MODEL.META_ARCH.GRADIENT_MUL = 0.1
_C.MODEL.META_ARCH.PRIOR_PROB = 0.01

# for retina based IoU Matcher
_C.MODEL.META_ARCH.BBOX_REG_WEIGHTS = 1.0
_C.MODEL.META_ARCH.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.META_ARCH.IOU_LABELS = [0, -1, 1]

_C.MODEL.META_ARCH.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.META_ARCH.FOCAL_LOSS_ALPHA = 0.25

# re-weight
_C.MODEL.META_ARCH.LOSS_CLS_WEIGHT = 1.0
_C.MODEL.META_ARCH.LOSS_LOC_INIT_WEIGHT = 0.5
_C.MODEL.META_ARCH.LOSS_LOC_REFINE_WEIGHT = 1.0

# for point set representation
_C.MODEL.META_ARCH.POINT_BASE_SCALE = 4
_C.MODEL.META_ARCH.NUM_POINTS = 9
_C.MODEL.META_ARCH.TRANSFORM_METHOD = "minmax"
_C.MODEL.META_ARCH.MOMENT_MUL = 0.01

_C.MODEL.META_ARCH.CENTER_SAMPLING_RADIUS = 0.0
_C.MODEL.META_ARCH.NORM_REG_TARGETS = False
_C.MODEL.META_ARCH.CENTERNESS_ON_LOC = False
_C.MODEL.META_ARCH.IOU_LOSS_TYPE = "iou"

# for inference
_C.MODEL.META_ARCH.SCORE_THRESH_TEST = 0.05
_C.MODEL.META_ARCH.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.META_ARCH.NMS_THRESH_TEST = 0.5

_C.MODEL.META_ARCH.PRE_NMS_TOP_N = 1000
_C.MODEL.META_ARCH.PRE_NMS_THRESH = 0.05

#for slenderness
_C.MODEL.META_ARCH.SLENDER_CENTERNESS = False


def get_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C
