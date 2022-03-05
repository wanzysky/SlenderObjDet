import fire
import ipdb
import torch
import torch.nn as nn

from detectron2.utils.logger import setup_logger
from slender_det.checkpoint import DetectionCheckpointer
from slender_det.config import get_cfg
from slender_det.modeling import build_model
from slender_det.modeling.backbone import build_backbone

# get cfg
output_dir = "./output/test_model"
setup_logger(output_dir, name="fvcore")
setup_logger(output_dir, name="slender_det")
setup_logger(output_dir)
cfg = get_cfg()
cfg.merge_from_file("../configs/ablation_studies/pointset/base_pvt_l_FPN_2x.yaml")

# get model
device = torch.device("cuda")
# model = build_model(cfg)

# get batch data
# data_loader = build_detection_train_loader(cfg)
# data_loader_iter = iter(data_loader)
# data = next(data_loader_iter)


def test_backbone():
    backbone = build_backbone(cfg).to(device)
    images = torch.empty((2, 3, 512, 512)).to(device)
    assert isinstance(backbone, nn.Module)

    outputs = backbone(images)

    ipdb.set_trace()


def test_load_pt():
    model = build_model(cfg)

    ipdb.set_trace()


if __name__ == "__main__":
    fire.Fire()
