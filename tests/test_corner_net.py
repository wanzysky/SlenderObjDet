import fire

import torch
import torch.nn as nn

from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer

import init_paths
from slender_det.config import get_cfg
from slender_det.modeling.backbone import build_backbone
from slender_det.modeling import build_model

# get cfg
cfg = get_cfg()
cfg.merge_from_file("configs/corner/Base-CornerNet.yaml")

# get model
device = torch.device("cuda")
model = build_model(cfg)

# get batch data
data_loader = build_detection_train_loader(cfg)
data_loader_iter = iter(data_loader)
data = next(data_loader_iter)


def test_backbone():
    backbone = build_backbone(cfg).to(device)
    images = torch.empty((2, 3, 512, 512)).to(device)
    assert isinstance(backbone, nn.Module)

    num = 0
    for module in backbone.modules():
        if isinstance(module, nn.Conv2d):
            num += 1

    print(num)
    outputs = backbone(images)

    import pdb
    pdb.set_trace()


def test_training():
    model.train()
    model(data[:2])

    import pdb
    pdb.set_trace()


def test_inference():
    if model is not None:
        model.eval()
        # DetectionCheckpointer(model).load("/data/exps/Detectron2/05-30_focs_R_50_FPN_1x/model_0084999.pth")
        # outs = model(data[:2])

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
