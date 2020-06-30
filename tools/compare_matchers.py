import fire

from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.modeling.matcher import Matcher

import init_paths
from slender_det.config import get_cfg
from slender_det.modeling.matchers import TopKMatcer


def setup(file):
    # get cfg
    cfg = get_cfg()
    cfg.merge_from_file(file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    return cfg


def main():
    raise NotImplementedError


if __name__ == '__main__':
    fire.Fire(main)
