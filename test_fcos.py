import torch
from slender_det.config import get_cfg
from slender_det.modeling.backbone import build_backbone
from slender_det.data import BorderMaskMapper
from detectron2.data import MetadataCatalog, build_detection_train_loader

cfg = get_cfg()
cfg.merge_from_file("configs/fcos/Base-Fcos.yaml")

# backbone = build_backbone(cfg)

from slender_det.modeling.meta_arch.fcos import FCOS, FCOSHead

device = torch.device("cuda")
model = FCOS(cfg).to(device)

model.train()
data_loader = build_detection_train_loader(cfg)
data_loader_iter = iter(data_loader)
data = next(data_loader_iter)

outs = model(data[:2])

import pdb
pdb.set_trace()
