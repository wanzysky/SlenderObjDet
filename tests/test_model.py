import fire
import ipdb
import torch

from detectron2.utils.logger import setup_logger
from slender_det.config import get_cfg
from slender_det.data import (
    build_detection_train_loader,
    get_detection_dataset_dicts,
    mappers,
)
from slender_det.modeling import build_model


def test_model(cfg_file):
    # get cfg
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.SOLVER.IMS_PER_BATCH = 1

    # get model
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    ipdb.set_trace()


def test_training(cfg_file):
    # get cfg
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.SOLVER.IMS_PER_BATCH = 2

    # get batch data
    data_loader = build_detection_train_loader(cfg)
    data_loader_iter = iter(data_loader)
    data = next(data_loader_iter)
    print(len(data))

    # get model
    device = torch.device("cuda")
    model = build_model(cfg).to(device)

    model.train()
    outs = model(data[:2])

    ipdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
