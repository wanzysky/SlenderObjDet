import ipdb
import time

from detectron2.config import get_cfg
from detectron2.data import build_detection_train_loader
from detectron2.utils.logger import setup_logger


def setup(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    setup_logger(name="detectron2")
    # cfg.DATASETS.TRAIN = ("coco_2017_val",)
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.MAX_ITER = 10

    data_loader = build_detection_train_loader(cfg)

    return cfg, data_loader


def main(cfg_file):
    cfg, data_loader = setup(cfg_file)
    data_loader_iter = iter(data_loader)

    data_time_list = []
    start = time.perf_counter()
    for idx, data in enumerate(data_loader_iter):
        data_time = time.perf_counter() - start
        print(data_time)
        data_time_list.append(data_time)
        time.sleep(0.5)
        if idx >= cfg.SOLVER.MAX_ITER:
            break
        start = time.perf_counter()


if __name__ == "__main__":
    main(cfg_file="detectron2/configs/Base-RCNN-C4.yaml")
