#print Copyrighs (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import os
from itertools import chain
from collections import defaultdict, OrderedDict

import numpy as np
import json
import cv2
import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from fvcore.common.file_io import PathManager
from tabulate import tabulate
from neupeak.utils import webcv2

from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.data.build import filter_images_with_few_keypoints
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import Boxes, BoxMode

classnames = [
"person",
"motorcycle",
"train",
"traffic light",
"parking meter",
"cat",
"sheep",
"bear",
"backpack",
"tie",
"skis",
"kite",
"skateboard",
"bottle",
"fork",
"bowl",
"sandwich",
"carrot",
"donut",
"couch",
"dining table",
"laptop",
"keyboard",
"oven",
"refrigerator",
"vase",
"hair drier",
"bicycle",
"airplane",
"truck",
"fire hydrant",
"bench",
"dog",
"cow",
"zebra",
"umbrella",
"suitcase",
"snowboard",
"baseball bat",
"surfboard",
"wine glass",
"knife",
"banana",
"orange",
"hot dog",
"cake",
"potted plant",
"toilet",
"mouse",
"cell phone",
"toaster",
"book",
"scissors",
"toothbrush",
"car",
"bus",
"boat",
"stop sign",
"bird",
"horse",
"elephant",
"giraffe",
"handbag",
"frisbee",
"sports ball",
"baseball glove",
"tennis racket",
"cup",
"spoon",
"apple",
"broccoli",
"pizza",
"chair",
"bed",
"tv",
"remote",
"microwave",
"sink",
"clock",
"teddy bear"]


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
    parser.add_argument("--input", required=True, help="JSON file produced by the model")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
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
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

    def output(vis, fname):
        if args.show:
            print(fname)
            webcv2.imshow("window", vis.get_image()[:, :, ::-1])
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
                img = per_image["image"].permute(1, 2, 0)
                if cfg.INPUT.FORMAT == "BGR":
                    img = img[:, :, [2, 1, 0]]
                else:
                    img = np.asarray(Image.fromarray(img, mode=cfg.INPUT.FORMAT).convert("RGB"))

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

        class_info = defaultdict(list)
        area_info = defaultdict(list)
        names = metadata.get('thing_classes', None)
        for dic in tqdm.tqdm(dicts):
            annos = dic.get("annotations", None)
            boxes = [BoxMode.convert(x["bbox"], x["bbox_mode"], BoxMode.XYWH_ABS) for x in annos]
            for i, box in enumerate(boxes):
                try:
                    ratio = min(box[2] / box[3], box[3] / box[2])
                except ZeroDivisionError:
                    ratio = 0
                if ratio > 20 or ratio < 0.05:
                    continue
                name = names[annos[i]['category_id']]
                class_info[name].append(ratio)
                area_info[name].append(box[3] * box[2])

        ratios = []
        ratio_names = []
        infos = [['category', 'mean_ratio', 'max_ratio', 'min_ratio', 'std_ratio', 'mean_area', 'max_area', 'min_area', 'std_area']]
        for label_name in classnames:
            ratio_list = class_info[label_name]
            ratios_a = np.array(ratio_list)
            area_a = np.array(area_info[label_name])
            if ratios_a.ndim == 0:
                infos.append([label_name, 0, 0, 0, 0])
                continue
            try:
                infos.append([label_name,
                    ratios_a.mean(), ratios_a.max(), ratios_a.min(), ratios_a.std(axis=0),
                    area_a.mean(), area_a.max(), area_a.min(), area_a.std(axis=0)])
            except:
                import ipdb
                ipdb.set_trace()
            ratios.append(ratio_list)
            ratio_names.append(label_name)

        print(tabulate(infos, tablefmt='github'))
        fig = plt.gcf()
        fig.set_size_inches(18.5, 10.5)
        plt.boxplot(ratios, labels=ratio_names, whis=[0, 10], showfliers=False)
        plt.xticks(np.arange(len(ratios)) + 1, ratio_names, rotation='vertical')
        plt.yscale = 'log'
        plt.show()
