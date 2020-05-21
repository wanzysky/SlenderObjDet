import logging
import copy

import cv2
import numpy as np
import torch
from fvcore.common.file_io import PathManager
from PIL import Image
from detectron2.data import DatasetMapper as D2Mapper
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.utils.logger import log_every_n_seconds

from concern.smart_path import smart_path
from utils.nori_redis import NoriRedis


class BorderMaskMapper(D2Mapper):
    """
    Inherited from DatasetMapper of Detectron2, but adding border masks to data dicts.
    """

    def __init__(
        self,
        cfg,
        mask_keys=["sizes"],
        is_train=True
    ):
        super().__init__(cfg, is_train=is_train)
        assert len(mask_keys) > 0
        self.mask_keys = mask_keys
        self.prepare_nori(cfg)
        self.is_train = is_train
        self.nori_redis = None

    def prepare_nori(self, cfg):
        self.use_nori = cfg.USE_NORI
        if not self.use_nori:
            return
        split_name = "val2017" if self.is_train else "val2017"

        self.image_fetcher = NoriRedis(
            cfg,
            smart_path(cfg.NORI_PATH).joinpath(split_name + ".nori").as_uri())
        self.sizes_fecher = NoriRedis(
            cfg,
            smart_path(cfg.NORI_PATH).joinpath(split_name + "_sizes.nori").as_uri())

    def masks_for_image(self, dataset_dict, transforms):
        image_name = smart_path(dataset_dict["file_name"]).name
        masks = dict()
        for key in self.mask_keys:
            data = self.sizes_fecher.fetch(image_name)
            image = np.fromstring(data, dtype=np.float32)

            if key == "sizes":
                image = image.reshape((dataset_dict["height"], dataset_dict["width"], 2))
                image = image.transpose(2, 0, 1)
                resize_tfm = transforms.transforms[0]
                assert isinstance(resize_tfm, T.ResizeTransform)
                ratio_h = resize_tfm.new_h / resize_tfm.h
                ratio_w = resize_tfm.new_w / resize_tfm.w
                image = np.stack(
                    [transforms.apply_image(image[0] * ratio_w),
                     transforms.apply_image(image[1] * ratio_h)],
                    axis=0)
            else:
                image = image.reshape((dataset_dict["height"], dataset_dict["width"]))
                image = transforms.apply_image(image)

            masks[key] = torch.as_tensor(np.ascontiguousarray(image.copy()))
            del image
        return masks

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file

        if self.use_nori:
            data = self.image_fetcher.fetch(dataset_dict["file_name"])
            image = cv2.imdecode(np.fromstring(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        else:
            image = utils.read_image(dataset_dict["file_name"], format=self.img_format)

        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms
        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
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
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        dataset_dict.update(self.masks_for_image(dataset_dict, transforms))
        for key in self.mask_keys:
            assert dataset_dict[key].shape[-2:] == dataset_dict["image"].shape[1:], dataset_dict[key].shape
        return dataset_dict
