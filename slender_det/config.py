from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

#-------------------------------------------------------------------------------#
# Required by SlenderDet
#-------------------------------------------------------------------------------#

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

def get_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C
