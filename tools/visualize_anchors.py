from slender_det.structures.masks import PolygonMasks
from utils.visualizer import Visualizer
from concern.support import any_of, between
from concern import webcv2
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
from detectron2.structures import Keypoints

import sys
sys.path.append("..")


def save(image, *names):
    path = os.path.join(*names)
    dirname = os.path.dirname(path)
    os.makedirs(dirname, exist_ok=True)
    cv2.imwrite(path, image)


def create_instances_with_anchor(predictions, image_size):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = (score > args.conf_threshold).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    box_ratio = (bbox[:, 0] - bbox[:, 2]) / (bbox[:, 1] - bbox[:, 3])
    bbox = BoxMode.convert(bbox, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    anchors = np.asarray([predictions[i]["anchor"] for i in chosen]).reshape(-1, 4)
    anchors = BoxMode.convert(anchors, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray([dataset_id_map(predictions[i]["category_id"]) for i in chosen])

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)
    ret.pred_classes = labels
    ret.anchors = Boxes(anchors)

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
#    parser.add_argument("--proposal", required=True, help="Pickle file storing proposals")
    parser.add_argument("--result", required=True, help="JSON file produced by the model")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--show", action="store_true", help="show imags via webcv")
    parser.add_argument("--oriented", action="store_true",
                        help="use oriented boxes for ratio computation")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--iou-threshold", default=0.05, type=float, help="confidence threshold")
    parser.add_argument("--multisave", action="store_true",
                        help="save visualization results in multiple forms")
    args = parser.parse_args()

    interest = set(['baseball bat', 'knife', 'bench'])

    logger = setup_logger()

    # load instance
    with PathManager.open(args.result, "r") as f:
        predictions = json.load(f)
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    # load dataset
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

    # split visualization dir by ratio
    ratios_ranges = {
        "0-0.2": [(0, 0.2), (5, float("inf"))],
        "0.2-0.3*": [(1. / 5, 1. / 3), (3, 5)],
        # "0.3*-1": [(1. / 3, 3)]
    }

    if args.multisave:
        for ratio_name in ratios_ranges.keys():
            os.makedirs(os.path.join(args.output, ratio_name), exist_ok=True)

    if args.show:
        scale = 1
    else:
        scale = 0.5

    ratio_counts = {key: 0 for key in ratios_ranges.keys()}

    bar = tqdm.tqdm(dicts)
    for dic in bar:
        img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        basename = ".".join(os.path.basename(dic["file_name"]).split(".")[:-1])

        vis = Visualizer(img, metadata, scale=0.5)
        dic["annotations"] = [x for x in dic["annotations"] if x["iscrowd"] == 0]
        segmentations = [x["segmentation"] for x in dic["annotations"]]
        segmentations = PolygonMasks(segmentations)
        ratios = segmentations.get_ratios(oriented=args.oriented)

        predictions = create_instances_with_anchor(pred_by_image[dic["image_id"]], img.shape[:2])
        if not len(predictions) > 0:
            continue
        grouped_gt = vis.group_by(dic["annotations"], ratios, ratios_ranges)

        visualized = False
        for range_name in ratios_ranges.keys():
            if not len(grouped_gt[range_name]) > 0:
                continue
            visualized =True

            vis = Visualizer(img, metadata, scale=scale)
            topk_boxes, topk_indices = vis.topk_iou_boxes(
                predictions.pred_boxes,
                Boxes([BoxMode.convert(x["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS) for x in grouped_gt[range_name]]
                      ))
            topk_indices = topk_indices.reshape((-1, ))
            # Transform indices to list since shape 1 tensors will be regarded as scalars.
            vis.draw_dataset_dict({"annotations": grouped_gt[range_name]})
            vis_boxes = vis.draw_instance_predictions(predictions[topk_indices.tolist()])

            if args.show:
                webcv2.imshow(basename + "-boxes@" + range_name, vis_boxes.get_image()[..., ::-1])
            else:
                save(vis_boxes.get_image()[..., ::-1], 
                     args.output, "boxes", basename + "@%s.jpg" %
                     range_name)

            vis_anchor = Visualizer(img, metadata)
            anchors = predictions.anchors.tensor[topk_indices]
            vis_anchor = vis_anchor.overlay_instances(
                boxes=anchors.reshape(-1, 4),
                labels=predictions.scores[topk_indices.reshape(-1).tolist()])

            if args.show:
                webcv2.imshow(basename + "-anchors@" + range_name,
                              vis_anchor.get_image()[..., ::-1])
            else:
                save(vis_anchor.get_image()[..., ::-1], 
                     args.output, "anchors", basename + "@%s.jpg" %
                     range_name)
            ratio_counts[range_name] += 1

        if not visualized:
            continue

        vis = Visualizer(img, metadata, scale=0.5)
        vis_gt = vis.draw_dataset_dict(dic)
        if args.show:
            webcv2.imshow(basename + '@gt', vis_gt.get_image()[..., ::-1])
        else:
            save(vis_gt.get_image()[..., ::-1], 
                 args.output, "gt", basename + ".jpg")

        vis = Visualizer(img, metadata, scale=0.5)
        vis_pred = vis.draw_instance_predictions(predictions)

        if args.show:
            webcv2.imshow(basename + '@pred', vis_pred.get_image()[..., ::-1])
        else:
            save(vis_pred.get_image()[..., ::-1], 
                 args.output, "pred", basename + ".jpg")

        if args.show:
            import random
            if random.random() < 0.1:
                webcv2.waitKey()
