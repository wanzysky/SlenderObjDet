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

# Only available in Brain++
import nori2
from ._setup import setup
from utils.nori_redis import NoriRedis


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--output-file", default="s3://detection/slender-det/coco/val2017.nori", help="path to output directory")
    parser.add_argument("--image_path", default=None, help="path to images")
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


def save(mask:np.ndarray, *file_names):
    assert len(file_names) > 0
    save_path = smart_path(os.path.join(file_names[0], *file_names[1:]))
    with save_path.open('wb') as writer:
        writer.write(mask.tostring())


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
        webcv2.imshow(dic["file_name"] + "-sizes", (sizes / sizes.max()).sum(-1) * 255)
        webcv2.waitKey()
    else:
        file_name = os.path.basename(dic["file_name"])
        save(borders, cfg.MASK_DIRECTORY, split_name, "borders", file_name)
        save(centers, cfg.MASK_DIRECTORY, split_name, "centers", file_name)
        save(sizes, cfg.MASK_DIRECTORY, split_name, "sizes", file_name)


if __name__ == '__main__':
    args = parse_args()

    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)

    split_name = getattr(cfg.DATASETS, args.source)[0]
    metadata = MetadataCatalog.get(split_name)

    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in getattr(cfg.DATASETS, args.source)]))
    if args.processes > 0:
        pool = Pool(args.processes)

    output_path = smart_path(args.output_file)
    if output_path.exists():
        print(output_path.as_uri, "exists!")
        import sys
        sys.exit()

    with nori2.open(output_path.as_uri(), "w") as writer:
        for dic in tqdm.tqdm(dicts):
            if args.image_path is None:
                file_path = smart_path(dic["file_name"])
            else:
                file_path = smart_path(args.image_path).joinpath(smart_path(dic["file_name"]).name)

            with file_path.open("rb") as reader:
                writer.put(reader.read(), filename=file_path.name)
    
    redis = NoriRedis(
        output_path.as_uri(),
        cfg.REDIS.HOST,
        cfg.REDIS.PORT,
        cfg.REDIS.DB)
    redis.sync()
    os.system("nori speedup --on " + output_path.as_uri())
