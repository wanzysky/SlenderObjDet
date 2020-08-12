#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# It is used in fig: examples in the paper.

import argparse
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances, pairwise_iou
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from concern import webcv2
from concern.support import ratio_of_polygon, ratio_of_bbox


def create_instances(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


def evaluate_box_proposal(predictions, dic, limit=100, threshold=0.5, aspect_ratio_range=(0, 1/3)):
    gt_overlaps = []
    num_pos = 0

    anno = dic["annotations"]
    new_dic = []
    gt_boxes = [
        BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        for obj in anno
    ]
    gt_aspect_ratios = [
        ratio_of_polygon(obj["segmentation"]) if not obj["iscrowd"] else ratio_of_bbox(obj["bbox"])
        for obj in anno
    ]
    gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
    gt_boxes = Boxes(gt_boxes)
    gt_aspect_ratios = torch.as_tensor(gt_aspect_ratios)

    if len(gt_boxes) == 0:
        return None

    predict_boxes = [
        BoxMode.convert(prediction['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
        for prediction in predictions
    ]
    predict_boxes = torch.as_tensor(predict_boxes).reshape(-1, 4)
    predict_boxes = Boxes(predict_boxes)

    valid_gt_inds = (gt_aspect_ratios >= aspect_ratio_range[0]) & \
                    (gt_aspect_ratios <= aspect_ratio_range[1])
    gt_boxes = gt_boxes[valid_gt_inds]

    if len(gt_boxes) == 0 or len(predictions) == 0:
        return None

    num_pos += len(gt_boxes)
    if limit is not None and len(predictions) > limit:
        predict_boxes = predict_boxes[:limit]

    overlaps = pairwise_iou(predict_boxes, gt_boxes)

    selected_gt = [anno[i] for i, bl in enumerate(valid_gt_inds) if bl]
    selected_pred = []
    _gt_overlaps = torch.zeros(len(gt_boxes))
    pred_classes = []
    for j in range(min(len(predictions), len(gt_boxes))):
        # find which proposal box maximally covers each gt box
        # and get the iou amount of coverage for each gt box
        max_overlaps, argmax_overlaps = overlaps.max(dim=0)

        # find which gt box is 'best' covered (i.e. 'best' = most iou)
        gt_ovr, gt_ind = max_overlaps.max(dim=0)
        assert gt_ovr >= 0
        # find the proposal box that covers the best covered gt box
        box_ind = argmax_overlaps[gt_ind]
        # record the iou coverage of this gt box
        _gt_overlaps[j] = overlaps[box_ind, gt_ind]

        overlaped_box_ind = overlaps[:, gt_ind] > threshold
        if overlaped_box_ind.sum() > 0:
            pred_classes += [predictions[i]["category_id"] for i, bl in enumerate(overlaped_box_ind) if bl]
            selected_pred += [predictions[i] for i, bl in enumerate(overlaped_box_ind) if bl]
        assert _gt_overlaps[j] == gt_ovr
        # mark the proposal box and the gt box as used
        overlaps[box_ind, :] = -1
        overlaps[:, gt_ind] = -1

    # append recorded iou coverage level
    gt_overlaps = _gt_overlaps
    gt_overlaps, _ = torch.sort(gt_overlaps)

    dic["annotations"] = selected_gt
    return selected_pred, dic


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--show", action="store_true", help="show imags via webcv")
    parser.add_argument("--conf-threshold", default=0.2, type=float, help="confidence threshold")
    args = parser.parse_args()

    logger = setup_logger()

    with PathManager.open(args.input, "r") as f:
        predictions = json.load(f)

    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dicts = list(DatasetCatalog.get(args.dataset))
    metadata = MetadataCatalog.get(args.dataset)
    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    elif "lvis" in args.dataset:
        # LVIS results are in the same format as COCO results, but have a different
        # mapping from dataset category id to contiguous category id in [0, #categories - 1]
        def dataset_id_map(ds_id):
            return ds_id - 1

    else:
        raise ValueError("Unsupported dataset: {}".format(args.dataset))

    os.makedirs(args.output, exist_ok=True)
    for dic in tqdm.tqdm(dicts):
        flag = False
        for ann in dic['annotations']:
            category = metadata.get('thing_classes')[ann['category_id']]


        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        selected = evaluate_box_proposal(pred_by_image[dic["image_id"]], dic)
        if selected is None:
            continue
        predictions, dic = selected
        predictions = create_instances(predictions, img.shape[:2])


        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_instance_predictions(predictions).get_image()

        vis = Visualizer(img, metadata)
        vis_gt = vis.draw_dataset_dict(dic).get_image()

        concat = np.concatenate((vis_pred, vis_gt), axis=1)
        
        # webcv2.imshow(basename, concat[:, :, ::-1])
        # webcv2.waitKey()

        cv2.imwrite(os.path.join(args.output, basename), concat[:, :, ::-1])
