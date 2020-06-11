import fire

import torch

from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer

import init_paths
from slender_det.config import get_cfg
from slender_det.data import BorderMaskMapper
from slender_det.modeling import build_model

# get cfg
cfg = get_cfg()
cfg.merge_from_file("configs/fcos_anchor/Base-FcosAnchor.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
print(cfg.MODEL.ANCHOR_GENERATOR.SIZES)

# get model
device = torch.device("cuda")
model = build_model(cfg).to(device)

# get batch data
data_loader = build_detection_train_loader(cfg)
data_loader_iter = iter(data_loader)
data = next(data_loader_iter)


def test_training():
    model.train()
    outs = model(data)

    import pdb
    pdb.set_trace()


def test_inference(path_to_resume="/data/exps/Detectron2/06-10_fcos_anchor_R_50_FPN_1x/model_0064999.pth"):
    model.eval()
    DetectionCheckpointer(model).load(path_to_resume)
    outs = model(data)

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
