import ipdb
import os
import json
import cv2
import s3path
from PIL import Image
import io
import itertools
import torch
import copy
from concern.smart_path import smart_path

from pycocotools.coco import COCO
from fvcore.common.file_io import PathManager
import detectron2.data.detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data.detection_utils import convert_image_to_rgb

from slender_det.evaluation.obj365 import _convert_rle_to_polygon
from slender_det.data import get_detection_dataset_dicts


def load_image_from_oss(path: s3path.S3Path, mode='rb', format=None):
    """

    Args:
        path:
        mode:
        format:

    Returns:

    """
    assert isinstance(path, s3path.S3Path)
    image = Image.open(io.BytesIO(path.open(mode=mode).read()))
    image = utils.convert_PIL_to_numpy(image, format)
    return image


def test_rle_to_polygon(res_file):
    obj365_anno_file = "datasets/obj365/annotations/objects365_val_20190423.json"
    obj_api = COCO(annotation_file=obj365_anno_file)

    with open(file=res_file, mode='rb') as f:
        predictions = torch.load(f)

    anns = list(itertools.chain(*[x["instances"] for x in predictions]))

    oss_root = "s3://detection/objects365_raw_data/objects365/val/"
    img_format = "BGR"

    for ann in anns:
        segm = ann['segmentation']
        image_dict = obj_api.imgs[ann['image_id']]
        file_path = os.path.join(oss_root, image_dict['file_name'])
        image = load_image_from_oss(smart_path(file_path), format=img_format)
        image = convert_image_to_rgb(image, format=img_format)
        v_rle = Visualizer(image, None)
        v_rle = v_rle.overlay_instances(
            masks=[segm],
        )
        v_rle.save('/data/tmp/{}_rle.png'.format(ann['image_id']))

        v_polygon = Visualizer(copy.deepcopy(image), None)
        v_polygon = v_polygon.overlay_instances(
            masks=[_convert_rle_to_polygon(segm)],
        )
        v_polygon.save('/data/tmp/{}_polygon.png'.format(ann['image_id']))

        ipdb.set_trace()


def test_dataset_dicts():
    dataset_dicts = get_detection_dataset_dicts(['coco_objects365_val_with_masks'], filter_empty=False, )
    oss_root = "s3://detection/"
    img_format = "BGR"
    for dataset_dict in dataset_dicts:
        file_path = os.path.join(oss_root, dataset_dict['file_name'])
        image = load_image_from_oss(smart_path(file_path), format=img_format)
        image = convert_image_to_rgb(image, format=img_format)
        anns = dataset_dict['annotations']

        masks = [ann['segmentation'] for ann in anns]

        v_gt = Visualizer(image, None)
        v_gt = v_gt.overlay_instances(
            masks=masks,
        )
        v_gt.save('/data/tmp/test_dd_{}.png'.format(dataset_dict['image_id']))

        ipdb.set_trace()


def test_dataloader():
    from slender_det.data import build_detection_test_loader
    from slender_det.config import get_cfg

    cfg = get_cfg()
    cfg.DATASETS.TEST = ("coco_objects365_val_with_masks",)

    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    data_iter = iter(data_loader)

    # data = next(data_iter)
    # ipdb.set_trace()

    for data in data_iter:
        for dataset_dict in data:
            img = dataset_dict['image']
            img = convert_image_to_rgb(img.permute(1, 2, 0), cfg.INPUT.FORMAT)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(
                boxes=dataset_dict["instances"].gt_boxes,
                masks=dataset_dict["instances"].gt_masks,
            )
            v_gt.save('/data/tmp/vis_coco_obj365_val_{}.png'.format(dataset_dict['image_id']))

            ipdb.set_trace()


if __name__ == '__main__':
    import fire

    fire.Fire()
