import ipdb

import os
import json
from concern.smart_path import smart_path

import detectron2.data.datasets


def vis_dataset_dicts():
    from slender_det.data import get_detection_dataset_dicts
    from pycocotools.coco import COCO
    from slender_det.data.datasets.builtin_meta import _map_obj365_to_coco80

    root = "s3://detection/objects365_raw_data/objects365/"

    coco_anno_file = 'datasets/coco/annotations/instances_val2017.json'
    obj365_anno_file = "datasets/obj365/annotations/objects365_val_20190423.json"

    coco_api = COCO(annotation_file=coco_anno_file)
    obj_api = COCO(annotation_file=obj365_anno_file)
    coco_obj_api = COCO(annotation_file='datasets/obj365/annotations/test_polygon.json')

    map_oc = _map_obj365_to_coco80()
    dataset_dicts = get_detection_dataset_dicts(['coco_objects365_val_with_masks'], filter_empty=False, )
    coco_dataset_dicts = get_detection_dataset_dicts(['coco_2017_val'], filter_empty=False, )
    ipdb.set_trace()


def vis_batch_data():
    from slender_det.data import build_detection_test_loader
    from slender_det.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.detection_utils import convert_image_to_rgb

    cfg = get_cfg()
    cfg.DATASETS.TEST = ("objects365_train",)

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
                # masks=dataset_dict["instances"].gt_masks,
            )
            v_gt.save('/data/tmp/vis_coco_obj365_{}.png'.format(dataset_dict['image_id']))

            ipdb.set_trace()


if __name__ == '__main__':
    import fire

    fire.Fire()
