import argparse
from itertools import chain

import tqdm

from detectron2.config import get_cfg
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_train_loader
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer

from concern.smart_path import smart_path
from concern import webcv2
from slender_det.structures.borders import BorderMasks


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
    parser.add_argument("--show", action="store_true", help="show output in a window")
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser.parse_args(in_args)

if __name__ == '__main__':
    args = parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup(args)
    dirname = args.output_dir
    output_dir = smart_path(dirname)
    if not output_dir.exists():
        output_dir.mkdir(exist_ok=True)

    metadata = MetadataCatalog.get(getattr(cfg.DATASETS, args.source)[0])

    dicts = list(chain.from_iterable([DatasetCatalog.get(k) for k in getattr(cfg.DATASETS, args.source)]))
    for dic in tqdm.tqdm(dicts):
        img = utils.read_image(dic["file_name"], "RGB")
        visualizer = Visualizer(img, metadata=metadata)
        vis = visualizer.draw_dataset_dict(dic)
        webcv2.imshow(dic["file_name"], vis.get_image()[:, :, ::-1])
        try:
            masks = BorderMasks([x['segmentation'] for x in dic['annotations']])
            mask = masks.border_masks(mask_size=img.shape[:2])
        except:
            import ipdb
            ipdb.set_trace()
        webcv2.imshow(dic["file_name"] + "-mask", mask * 255)
        # webcv2.waitKey()
