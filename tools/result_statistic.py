import io
import contextlib
import os
from collections import defaultdict

import tqdm
import cv2
import pickle
import torch
import numpy as np
from detectron2.structures import Boxes, BoxMode, pairwise_iou, Instances
from detectron2.engine import default_argument_parser, default_setup
from detectron2.utils.logger import create_small_table
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import print_csv_format

from slender_det.structures.masks import PolygonMasks
from slender_det.evaluation.coco_evaluation import COCOEvaluator
from slender_det.evaluation.coco import COCO
from slender_det.config import get_cfg
from concern.smart_path import smart_path

from slender_det.data.mappers import load_image_from_oss

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    print(args.config_file)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def reconstruct_ann(dic):
    result = []
    for obj in dic:
        ratio = COCO.compute_ratio(obj)
        obj["ratio"] = ratio
        result.append(obj)

    return result


def create_instances(prediction, image_size):
    ret = Instances(image_size)

    scores = []
    pred_boxes = []
    pred_classes = []
    for instance in prediction["instances"]:
        scores.append(instance["score"])
        pred_boxes.append(instance["bbox"])
        pred_classes.append(instance["category_id"])

    scores = np.asarray(scores)
    pred_boxes = np.asarray(pred_boxes).reshape(-1, 4)
    pred_boxes = BoxMode.convert(pred_boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)

    labels = np.asarray(pred_classes)

    ret.scores = scores
    ret.pred_boxes = Boxes(pred_boxes)
    ret.pred_classes = labels

    return ret



def main():
    parser = default_argument_parser()
    parser.add_argument("--prediction", help="predictions_file_path")
    args = parser.parse_args()
    cfg = setup(args)
    with smart_path(args.prediction).open("rb") as fp:
        buf = io.BytesIO(fp.read())
        predictions = torch.load(buf)
    pred_by_image = defaultdict(list)
    for p in predictions:
        pred_by_image[p["image_id"]].append(p)

    dataset = cfg.DATASETS.TEST[0]

    metadata = MetadataCatalog.get(dataset)

    if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):

        def dataset_id_map(ds_id):
            return metadata.thing_dataset_id_to_contiguous_id[ds_id]

    output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator = COCOEvaluator(dataset, cfg, False, output_folder)
    evaluator.reset()

    dicts = list(DatasetCatalog.get(dataset))

    count = 0
    
    for dic in tqdm.tqdm(dicts):
        assert len(pred_by_image[dic["image_id"]]) == 1

        prediction = pred_by_image[dic["image_id"]][0]
        file_path = dic['file_name']
        file_path = os.path.join(cfg.DATALOADER.OSS_ROOT, file_path)
        img = load_image_from_oss(smart_path(file_path), format = cfg.INPUT.FORMAT)
        #img = cv2.imread(dic["file_name"], cv2.IMREAD_COLOR)[:, :, ::-1]
        prediction = create_instances(prediction, img.shape[:2])
        # Push an image
        dic["annotations"] = reconstruct_ann(dic["annotations"])
        evaluator.process([dic], [{"instances": prediction}])
        count += 1

    result = evaluator.evaluate()
    prediction_path = smart_path(args.prediction)
    save_path = prediction_path.parent.joinpath(prediction_path.stem + ".pkl")
    with save_path.open("wb") as writer:
        pickle.dump(result, writer)
    print_csv_format(result)

if __name__ == '__main__':
    main()
