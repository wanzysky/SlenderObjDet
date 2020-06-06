import argparse
import pickle
import json
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger

import sys
sys.path.append("..")
from utils.visualizer import Visualizer
from concern import webcv2


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--proposal", required=True, help="Pickle file storing proposals")
#    parser.add_argument("--result", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--show", action="store_true", help="show imags via webcv")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--iou-threshold", default=0.05, type=float, help="confidence threshold")
    args = parser.parse_args()

    interest = set(['baseball bat', 'knife', 'bench'])

    logger = setup_logger()
    with PathManager.open(args.proposal, "rb") as f:
        proposals = pickle.load(f)
    proposal_by_image = defaultdict(list)
    for image_id, bboxes, objectness_logits in zip(proposals['ids'], proposals['boxes'], proposals['objectness_logits']):
        for bbox, objectness_logit in zip(bboxes, objectness_logits):
            proposal_by_image[image_id].append({'bbox': bbox, 'score': 1 / (1 + np.exp(objectness_logit))})
    
    anchor_by_image = defaultdict(list)
    for image_id, anchors, objectness_logits in zip(proposals['ids'], proposals['anchors'], proposals['objectness_logits']):
        for anchor, objectness_logit in zip(anchors, objectness_logits):
            anchor_by_image[image_id].append({'bbox': anchor, 'score': 1 / (1 + np.exp(objectness_logit))})
    
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
            if category in interest:
                flag = True
        if not flag:
            continue
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = os.path.basename(dic["file_name"])

        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_proposals_separately(proposal_by_image[dic["image_id"]], img.shape[:2], args.conf_threshold)
        concat_proposal = vis.smart_concatenate(vis_pred, min_side=1960)
        
        vis = Visualizer(img, metadata)
        vis_pred = vis.draw_proposals_separately(anchor_by_image[dic["image_id"]], img.shape[:2], args.conf_threshold)
        concat_anchor = vis.smart_concatenate(vis_pred, min_side=1960)
        if args.show:
            webcv2.imshow(basename + ' - Press D for details', concat_proposal[:, :, ::-1])
            key = webcv2.waitKey()
            if key == 100: # 'd'
                webcv2.imshow(basename, concat[:, :, ::-1])
                webcv2.waitKey()
        else:
            cv2.imwrite(os.path.join(args.output, basename) + '-proposals.jpg', concat_proposal[:, :, ::-1])
            cv2.imwrite(os.path.join(args.output, basename) + '-anchors.jpg', concat_anchor[:, :, ::-1])