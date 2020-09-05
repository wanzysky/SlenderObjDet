import ipdb

import torch

from detectron2.utils.visualizer import Visualizer

from slender_det.config import get_cfg
from slender_det.data import build_detection_train_loader
from slender_det.modeling import build_model


def load_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    return cfg


def main(cfg_file):
    print(cfg_file)
    cfg = load_cfg(cfg_file)
    data_loader = build_detection_train_loader(cfg)

    device = torch.device("cuda")
    model = build_model(cfg).to(device)

    for data in data_loader:
        ipdb.set_trace()


if __name__ == '__main__':
    import fire

    fire.Fire(main)
