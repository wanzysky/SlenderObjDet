import ipdb
import os
import copy
import numpy as np

import torch

from detectron2.utils.visualizer import Visualizer, _create_text_labels, ColorMode, mplc
import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.structures import pairwise_iou

from slender_det.config import get_cfg
from slender_det.data.mappers import DatasetMapper
from slender_det.data import build_detection_test_loader, MetadataCatalog
from slender_det.modeling import build_model
from concern.support import ratio_of_polygon


def detector_postprocess(results):
    """
    Resize the output instances.
    The input images are often resized when entering an object detector.
    As a result, we often need the outputs of the detector in a different
    resolution from its inputs.

    This function will resize the raw outputs of an R-CNN detector
    to produce outputs according to the desired output resolution.

    Args:
        results (Instances): the raw outputs from the detector.
            `results.image_size` contains the input image resolution the detector sees.
            This object might be modified in-place.
        output_height, output_width: the desired output resolution.

    Returns:
        Instances: the resized output from the model, based on the output resolution
    """
    output_boxes = results.pred_boxes
    output_boxes.clip(results.image_size)

    results = results[output_boxes.nonempty()]

    return results


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


class OffsetsVisualizer(Visualizer):

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        print("draw offsets")
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        points = predictions.pred_points if predictions.has("pred_points") else None
        offsets = predictions.pred_offsets if predictions.has("pred_offsets") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))

        # if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
        if self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            assigned_colors=colors,
            alpha=alpha,
        )

        if isinstance(offsets, torch.Tensor):
            offsets = offsets.numpy()
        if isinstance(points, torch.Tensor):
            points = points.numpy()

        num_instances = len(points)
        for i in range(num_instances):
            color = colors[i]
            point = points[i]
            self.draw_circle(point, color=mplc.to_rgba("red"), radius=5)

            offset = offsets[i].reshape(-1, 2)
            point = np.repeat(point[None, :], axis=0, repeats=offset.shape[0])
            coordinates = point - offset[:, [1, 0]]
            for coord in coordinates.tolist():
                self.draw_circle(coord, color=color)

        return self.output


def load_cfg(cfg_file):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_file)
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.DATASETS.TEST = ("coco_2017_val",)
    cfg.DATASETS.TEST = ("coco_objects365_val_with_masks",)
    cfg.MODEL.META_ARCH.NAME = 'PointSetVISHead'
    return cfg


def forward(model, batched_inputs):
    images = model.preprocess_image(batched_inputs)
    features = model.backbone(images.tensor)
    features = [features[f] for f in model.head.in_features]

    results = model.head(images, features)

    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes):
        r = detector_postprocess(results_per_image)
        processed_results.append({"instances": r})
    return processed_results


def forward_rpd(model, batched_inputs):
    model.eval()

    images = model.preprocess_image(batched_inputs)
    results = model(batched_inputs)

    processed_results = []
    for results_per_image, input_per_image, image_size in zip(
            results, batched_inputs, images.image_sizes):
        r = detector_postprocess(results_per_image)
        processed_results.append({"instances": r})
    return processed_results


def plot_offsets(batched_inputs, batched_results, cfg, save_dir=None):
    for input, result in zip(batched_inputs, batched_results):
        file_name = input['file_name'].split('/')[-1]
        img = input["image"]
        img = utils.convert_image_to_rgb(img.permute(1, 2, 0), cfg.INPUT.FORMAT)
        gt_instances = input['instances']

        # choose only slender object for visualization
        rars = [ratio_of_polygon(polygon) for polygon in gt_instances.gt_masks.polygons]
        rars = torch.tensor(rars)
        gt_instances_vis = gt_instances[rars < 0.3]
        if len(gt_instances_vis) <= 0:
            continue

        predictions = result['instances'].to(torch.device('cpu'))
        predictions = predictions[predictions.scores > 0.5]

        # choose only slender predictions
        if len(predictions) <= 0:
            continue

        iou = pairwise_iou(gt_instances_vis.gt_boxes, predictions.pred_boxes)
        value, idxs = torch.max(iou, dim=1)
        valid_idxs = idxs[value > 0.5]
        predictions_vis = predictions[valid_idxs]
        if len(predictions_vis) <= 0:
            continue

        v_pred = OffsetsVisualizer(img, metadata=MetadataCatalog.get(cfg.DATASETS.TEST[0]))

        v_pred = v_pred.draw_instance_predictions(predictions_vis)
        if save_dir is not None:
            v_pred.save(os.path.join(save_dir, file_name))


def main(cfg_file, save_dir="/data/tmp/offsets_slender_unsup"):
    print(cfg_file)
    cfg = load_cfg(cfg_file)

    from detectron2.data.build import \
        get_detection_dataset_dicts, DatasetFromList, MapDataset, InferenceSampler, trivial_batch_collator
    dataset_dicts = get_detection_dataset_dicts(
        [cfg.DATASETS.TEST[0]],
        filter_empty=False,
    )[5000:]
    dataset = DatasetFromList(dataset_dicts)
    mapper = Mapper(cfg, is_train=False)

    # data_loader = build_detection_test_loader(cfg, dataset_name=cfg.DATASETS.TEST[0], mapper=mapper)
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
    device = torch.device("cuda")
    model = build_model(cfg).to(device)

    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
    )

    assert isinstance(model, torch.nn.Module)
    model.eval()

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # 1. select slender boxes
            # 2. compute offsets for slender boxes
            if cfg.MODEL.META_ARCHITECTURE == "AblationMetaArch":
                results = forward(model, data)
            elif cfg.MODEL.META_ARCHITECTURE == "RepPointsVISDetector":
                results = forward_rpd(model, data)
            else:
                raise ValueError("{}".format(cfg.MODEL.META_ARCHITECTURE))

            plot_offsets(data, results, cfg, save_dir=save_dir)

            if i >= 200:
                break


if __name__ == '__main__':
    import fire

    fire.Fire(main)
