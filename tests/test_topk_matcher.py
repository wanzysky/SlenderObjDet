import fire

import torch

from detectron2.data import build_detection_train_loader
from detectron2.modeling.backbone import build_backbone
from detectron2.structures import Boxes, ImageList, pairwise_iou
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.matcher import Matcher

import init_paths
from slender_det.config import get_cfg
from slender_det.modeling.matchers import TopKMatcher

device = torch.device("cuda")


def setup(file):
    # get cfg
    cfg = get_cfg()
    cfg.merge_from_file(file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    # get data loader iter
    data_loader = build_detection_train_loader(cfg)
    data_loader_iter = iter(data_loader)
    batched_inputs = next(data_loader_iter)

    # build anchors
    backbone = build_backbone(cfg).to(device)
    images = [x["image"].to(device) for x in batched_inputs]
    images = ImageList.from_tensors(images, backbone.size_divisibility)
    features = backbone(images.tensor.float())

    input_shape = backbone.output_shape()
    in_features = cfg.MODEL.RPN.IN_FEATURES
    anchor_generator = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
    anchors = anchor_generator([features[f] for f in in_features])
    anchors = Boxes.cat(anchors).to(device)

    # build matcher
    raw_matcher = Matcher(
        cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
    )
    matcher = TopKMatcher(cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, 9)

    return cfg, data_loader_iter, anchors, matcher, raw_matcher


def test(cfg, data_loader_iter, anchors, matcher, raw_matcher):
    batched_inputs = next(data_loader_iter)
    gt_instances = [x["instances"].to(device) for x in batched_inputs]
    gt_boxes = [x.gt_boxes for x in gt_instances]
    image_sizes = [x.image_size for x in gt_instances]
    del gt_instances

    for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
        match_quality_matrix = pairwise_iou(gt_boxes_i, anchors)
        raw_matched_idxs, raw_gt_labels_i = raw_matcher(match_quality_matrix)
        matched_idxs, gt_labels_i = matcher(match_quality_matrix)

        import pdb
        pdb.set_trace()


def main(file):
    cfg, data_loader_iter, anchors, matcher, raw_matcher = setup(file)
    test(cfg, data_loader_iter, anchors, matcher, raw_matcher)


if __name__ == '__main__':
    fire.Fire(main)
