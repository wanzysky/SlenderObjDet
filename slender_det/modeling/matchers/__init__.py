from .rep_matcher import rep_points_match, nearest_point_match, inside_match
from detectron2.modeling.matcher import Matcher
from .topk_matcher import TopKMatcher
from .hungarian_matcher import HungarianMatcher

MATCHER_TYPES = ["Origin", "TopK"]


def build_matcher(cfg):
    type = cfg.MODEL.RPN.MATCHER.TYPE
    assert type in MATCHER_TYPES, "Matcher Type doesn't exist!" \
                                  "Expected one in {}," \
                                  "But got {}".format(MATCHER_TYPES, type)
    if type == "Origin":
        return Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
    elif type == "TopK":
        return TopKMatcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, cfg.MODEL.RPN.MATCHER.TOPK
        )
    else:
        raise ValueError("Unknown Matcher type: {}".format(type))
