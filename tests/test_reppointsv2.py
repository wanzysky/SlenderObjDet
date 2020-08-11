import fire

import torch

from detectron2.data import MetadataCatalog, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer

import init_paths
from slender_det.config import get_cfg
from slender_det.modeling import build_model

# get cfg
cfg = get_cfg()
cfg.merge_from_file("configs/rep-points/rep_pointsv2_R_50_FPN_1x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2

# get model
device = torch.device("cuda")
model = build_model(cfg).to(device)

# get batch data
data_loader = build_detection_train_loader(cfg)
data_loader_iter = iter(data_loader)
data = next(data_loader_iter)


def test_training():
    model.train()
    outs = model(data[:2])

    import pdb
    pdb.set_trace()


def test_inference():
    model.eval()
    # DetectionCheckpointer(model).load("/data/exps/Detectron2/05-30_focs_R_50_FPN_1x/model_0084999.pth")
    outs = model(data[:2])

    import pdb
    pdb.set_trace()


if __name__ == '__main__':
    fire.Fire()
