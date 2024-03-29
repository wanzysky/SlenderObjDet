#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import init_paths
import os
from concern import webcv2
import numpy as np
import cv2
import tqdm
from PIL import Image

from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from slender_det.config import get_cfg
from slender_det.data import MetadataCatalog, build_detection_train_loader


def setup(args):
    cfg = get_cfg()
    if args.config_file:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument("--speed", action="store_true", help="test speed of the loader")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


if __name__ == "__main__":
    args = parse_args()
    setup_logger(name="fvcore")
    setup_logger(name="slender_det")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            webcv2.imshow("window", vis.get_image()[:, :, ::-1])
            webcv2.waitKey()

    train_data_loader = build_detection_train_loader(cfg)
    import ipdb; ipdb.set_trace()
    for batch in tqdm.tqdm(train_data_loader):
        if args.speed:
            continue
        for per_image in batch:
            # Pytorch tensor is in (C, H, W) format
            img = per_image["image"].permute(1, 2, 0)
            if cfg.INPUT.FORMAT == "BGR":
                img = img[:, :, [2, 1, 0]]
            else:
                img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

            visualizer = Visualizer(img, metadata=metadata, scale=1.0)
            target_fields = per_image["instances"].get_fields()
            labels = [metadata.thing_classes[i] for i in target_fields["gt_classes"]]
            vis = visualizer.overlay_instances(
                labels=labels,
                boxes=target_fields.get("gt_boxes", None),
                masks=target_fields.get("gt_masks", None),
                keypoints=target_fields.get("gt_keypoints", None),
            )
            output(vis, str(per_image["image_id"]) + ".jpg")
