import argparse
from itertools import chain
from multiprocessing import Pool
import os

import tqdm
import cv2
import numpy as np

from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer

from concern.smart_path import smart_path
from concern import webcv2
from utils import encode_image
from slender_det.structures.borders import BorderMasks
from slender_det.config import get_cfg


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
    parser.add_argument("--output-dir", default="s3://detection/slender-det/coco/border-maps/", help="path to output directory")
    parser.add_argument(
        "--source",
        choices=["TRAIN", "TEST"],
        required=True,
        default="TEST",
        help="Training or test data for generation.")
    parser.add_argument("--processes", type=int, default=0, help="show output in a window")
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)


def save(mask, *file_names):
    assert len(file_names) > 0
    save_path = smart_path(os.path.join(file_names[0], *file_names[1:]))
    with save_path.open('wb') as writer:
        writer.write(encode_image(mask))

def process_single(metadata, dic, args):
    masks = BorderMasks([x['segmentation'] for x in dic['annotations'] if not isinstance(x['segmentation'], dict)])
    img = utils.read_image(dic["file_name"], "RGB")
    borders, centers, sizes = masks.masks(mask_size=img.shape[:2])

    if args.show:
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_dataset_dict(dic)
        webcv2.imshow(dic["file_name"], vis.get_image()[:, :, ::-1])
        webcv2.imshow(dic["file_name"] + "-border", borders * 255)
        webcv2.imshow(dic["file_name"] + "-centers", centers * 255)
        webcv2.imshow(dic["file_name"] + "-sizes", (1 - sizes / sizes.max()) * 255)
        webcv2.waitKey()
    else:
        save(borders, cfg.MASK_DIRECTORY, split_name, "borders", dic["file_name"])
        save(centers, cfg.MASK_DIRECTORY, split_name, "centers", dic["file_name"])
        save(sizes, cfg.MASK_DIRECTORY, split_name, "sizes", dic["file_name"])


if __name__ == '__main__':
    args = parse_args()
    assert not args.show or args.processes < 1

    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    dirname = args.output_dir
    output_dir = smart_path(dirname)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    split_name = getattr(cfg.DATASETS, args.source)[0]
    metadata = MetadataCatalog.get(split_name)

    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in getattr(cfg.DATASETS, args.source)]))
    if args.processes > 0:
        pool = Pool(args.processes)

    bar = tqdm.tqdm(unit_divisor=len(dicts))
    for dic in dicts:
        if args.processes > 0:
            pool.apply_async(process_single, (metadata, dic, args), callback=lambda x: bar.update())
        else:
            process_single(metadata, dic, args)

    pool.close()
    pool.join()
