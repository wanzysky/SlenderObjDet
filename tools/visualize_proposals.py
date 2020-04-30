import argparse
import pickle
import numpy as np
import os
from collections import defaultdict
import cv2
import tqdm
from fvcore.common.file_io import PathManager
from neupeak.utils import webcv2

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import Boxes, BoxMode, Instances
from detectron2.utils.logger import setup_logger

from utils.visualizer import Visualizer


def create_instances(predictions, image_size, chosen):
    ret = Instances(image_size)

    score = np.asarray([x["score"] for x in predictions])
    chosen = ((score > args.conf_threshold) * chosen).nonzero()[0]
    score = score[chosen]
    bbox = np.asarray([predictions[i]["bbox"] for i in chosen]).reshape(-1, 4)
    bbox = BoxMode.convert(bbox, BoxMode.XYXY_ABS, BoxMode.XYXY_ABS)

    ret.scores = score
    ret.pred_boxes = Boxes(bbox)

    try:
        ret.pred_masks = [predictions[i]["segmentation"] for i in chosen]
    except KeyError:
        pass
    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that visualizes the json predictions from COCO or LVIS dataset."
    )
    parser.add_argument("--input", required=True, help="Pickle file storing proposals")
    parser.add_argument("--output", required=True, help="output directory")
    parser.add_argument("--dataset", help="name of the dataset", default="coco_2017_val")
    parser.add_argument("--conf-threshold", default=0.5, type=float, help="confidence threshold")
    parser.add_argument("--iou-threshold", default=0.05, type=float, help="confidence threshold")
    args = parser.parse_args()

    interest = set(['baseball bat', 'knife'])

    logger = setup_logger()

    with PathManager.open(args.input, "rb") as f:
        predictions = pickle.load(f)

    pred_by_image = defaultdict(list)
    for image_id, bboxes, objectness_logits in zip(predictions['ids'], predictions['boxes'], predictions['objectness_logits']):
        for bbox, objectness_logit in zip(bboxes, objectness_logits):
            pred_by_image[image_id].append({'bbox': bbox, 'score': 1 / (1 + np.exp(objectness_logit)), 'category_id': 62})

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

        vis_pred = vis.draw_proposals_separately(pred_by_image[dic["image_id"]], img.shape[:2], args.conf_threshold)

        vis = Visualizer(img, metadata)
        vis_pred.append(vis.draw_dataset_dict(dic).get_image())

        concat = vis.smart_concatenate(vis_pred, out_shape=(2048, 2048))
        webcv2.imshow(os.path.join(args.output, basename), concat[:, :, ::-1])
        webcv2.waitKey()
