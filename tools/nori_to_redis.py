import sys
import argparse

from fire import Fire
from detectron2.utils.logger import setup_logger
from utils.nori_redis import NoriRedis

from ._setup import setup


def parse_args(in_args=None):
    parser = argparse.ArgumentParser(description="Visualize ground-truth data")
    parser.add_argument("--config-file", metavar="FILE", help="path to config file")
    parser.add_argument("--nori", default="s3://detection/slender-det/coco/val2017.nori", help="path of nori")
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

    redis = NoriRedis(
        cfg,
        args.nori)
    redis.sync()
