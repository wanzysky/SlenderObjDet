import ipdb

import os
import copy
import json
from concern.smart_path import smart_path
import numpy as np
import torch
import detectron2.data.datasets
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from slender_det.data.mappers import DatasetMapper


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


class Mapper(DatasetMapper):
    def __init__(self, cfg, is_train: bool = True):
        super().__init__(cfg, is_train)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = self.read_image(dataset_dict["file_name"])
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.StandardAugInput(image, sem_seg=sem_seg_gt)
        transforms = aug_input.apply_augmentations(self.augmentations)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # keep segmentation mask
                # if not self.use_instance_mask:
                #     anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.instance_mask_format
            )

            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict


def vis_batch_data():
    from slender_det.data import build_detection_test_loader
    from slender_det.config import get_cfg
    from detectron2.utils.visualizer import Visualizer
    from detectron2.data.detection_utils import convert_image_to_rgb

    cfg = get_cfg()
    cfg.DATASETS.TEST = ("coco_objects365_val_with_masks",)
    # cfg.DATASETS.TEST = ("coco_2017_val",)

    from detectron2.data.build import \
        get_detection_dataset_dicts, DatasetFromList, MapDataset, InferenceSampler, trivial_batch_collator
    dataset_dicts = get_detection_dataset_dicts(
        [cfg.DATASETS.TEST[0]],
        filter_empty=False,
    )[5000:]
    print(len(dataset_dicts))

    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, is_train=False)

    dataset = MapDataset(dataset, mapper)
    sampler = InferenceSampler(len(dataset))
    # Always use 1 image per worker during inference since this is the
    # standard when reporting inference time in papers.
    batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, 1, drop_last=False)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        batch_sampler=batch_sampler,
        collate_fn=trivial_batch_collator,
    )

    data_iter = iter(data_loader)

    # data = next(data_iter)
    # ipdb.set_trace()

    for data in data_iter:
        for dataset_dict in data:
            ipdb.set_trace()
            img = dataset_dict['image']
            img = convert_image_to_rgb(img.permute(1, 2, 0), cfg.INPUT.FORMAT)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(
                boxes=dataset_dict["instances"].gt_boxes,
                masks=dataset_dict["instances"].gt_masks,
            )
            v_gt.save('/data/tmp/vis_coco_{}.png'.format(dataset_dict['image_id']))

            ipdb.set_trace()


if __name__ == '__main__':
    import fire

    fire.Fire()
