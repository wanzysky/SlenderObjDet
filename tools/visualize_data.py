#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
import io
from itertools import chain
import cv2
import tqdm
from PIL import Image

from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer

from slender_det.config import get_cfg
from slender_det.evaluation.coco import COCO
from concern import webcv2
from slender_det.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from concern.smart_path import smart_path
from slender_det.structures.masks import PolygonMasks


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument(
        "--source",
        choices=["annotation", "dataloader"],
        required=True,
        help="visualize the annotations or the data loader (with pre-processing)",
    )
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-dir", default="./", help="path to output directory")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    dirname = args.output_dir
    os.makedirs(dirname, exist_ok=True)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            webcv2.imshow(fname, vis.get_image()[:, :, ::-1])
            webcv2.waitKey()
        else:
            filepath = os.path.join(dirname, fname)
            print("Saving to {} ...".format(filepath))
            vis.save(filepath)

    scale = 2.0 if args.show else 1.0
    if args.source == "dataloader":
        train_data_loader = build_detection_train_loader(cfg)
        for batch in train_data_loader:
            for per_image in batch:
                # Pytorch tensor is in (C, H, W) format
                img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
                img = utils.convert_image_to_rgb(img, cfg.INPUT.FORMAT)

                visualizer = Visualizer(img, metadata=metadata, scale=scale)
                target_fields = per_image["instances"].get_fields()
                labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
                vis = visualizer.overlay_instances(
                    labels=labels,
                    boxes=target_fields.get("gt_boxes", None),
                    masks=target_fields.get("gt_masks", None),
                    keypoints=target_fields.get("gt_keypoints", None),
                )
                output(vis, str(per_image["image_id"]) + ".jpg")
    else:
        dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in cfg.DATASETS.TEST]))
        if cfg.MODEL.KEYPOINT_ON:
            dicts = filter_images_with_few_keypoints(dicts, 1)

        all_count = 0
        empty_count = 0
        problematic_count = 0
        for dic in tqdm.tqdm(dicts):
            for obj in dic["annotations"]:
                all_count += 1
                if not "segmentation" in obj or len(obj["segmentation"]) == 0:
                    empty_count += 1
                    continue
                try:
                    segmentations = PolygonMasks([obj["segmentation"]])
                    ratio = segmentations.get_ratios(oriented=True)[0]
                    print(ratio)
                except:
                    problematic_count += 1
                ratio = COCO.compute_ratio(obj)

            img = utils.convert_PIL_to_numpy(
                    Image.open(io.BytesIO(smart_path(
                        "s3://detection/" + dic["file_name"]).read_bytes())),
                    "RGB")
            visualizer = Visualizer(img, metadata=metadata, scale=scale)
            vis = visualizer.draw_dataset_dict(dic)
            output(vis, os.path.basename(dic["file_name"]))

        print("all", all_count)
        print("empty", empty_count)
        print("problematic", problematic_count)
