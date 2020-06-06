from detectron2.modeling.meta_arch import build_model

from .pnwm import ProposalNetworkWithMasks
from .pvrcnn import ProposalVisibleRCNN
from .fcos import FCOS
from .retinanetwa import RetinaNetWithAnchor
from .fcoswa import FCOSWithAnchor
from .pvrcnnwa import ProposalVisibleRCNNWithAnchor
from .rpd import RepPointsDetector
