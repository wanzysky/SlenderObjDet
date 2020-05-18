from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg as d2config

_C = d2config()

#-------------------------------------------------------------------------------#
# Required by SlenderDet
#-------------------------------------------------------------------------------#

_C.MASK_DIRECTORY = "s3://detection/datasets/coco/masks/"

_C.MODEL.PROPOSAL_GENERATOR.IN_FEATURES = ["p5"]
_C.MODEL.PROPOSAL_GENERATOR.NUM_POINTS = 9
_C.MODEL.PROPOSAL_GENERATOR.SIZES = [8, 16, 32, 64, 128]

def get_cfg() -> CN:
    """
    Get a copy of the default config.

    Returns:
        a detectron2 CfgNode instance.
    """
    return _C
