import datetime
import json
import os
from concern import webcv2
from fire import Fire
from itertools import chain
import numpy as np
import cv2
import tqdm
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from slender_det.evaluation.coco import COCO


def main(input_path, output_path, show=False):
    coco = COCO(input_path)

    if show:
        data_dir = "datasets/coco/train2017/"
        dicts = DatasetCatalog.get("coco_2017_train")
        metadata = MetadataCatalog.get("coco_2017_train")
        for dic in dicts:
            for ann in dic["annotations"]:
                coco.compute_rbox(ann)
                ann["bbox"] = ann["rbox"]
                ann["bbox_mode"] = ann["rbox_mode"]
                print(ann["bbox"])

            image_path = dic["file_name"]
            img = utils.convert_PIL_to_numpy(
                    Image.open(image_path), "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=1)
            vis = visualizer.draw_dataset_dict(dic)
            webcv2.imshow(image_path+"bbox", vis.get_image()[:, :, ::-1])
            webcv2.waitKey()

    skip = False
    if "val" in input_path:
        skip = True

    anns = []
    for ann_id, ann in tqdm.tqdm(coco.anns.items()):
        if skip and ann["iscrowd"]:
            continue
        ann = coco.compute_rbox(ann)
        ann["bbox"] = ann["rbox"]
        anns.append(ann)

    info = dict(date_created=str(datetime.datetime.now()),
                description="Rbox version of {}.".format(
                    input_path))
    coco_dict = dict(
        info=info,
        categories=coco.dataset["categories"],
        annotations=anns,
        images=coco.dataset["images"],
        license=None)

    with open(output_path, "wt") as writer:
        json.dump(coco_dict, writer)


Fire(main)
