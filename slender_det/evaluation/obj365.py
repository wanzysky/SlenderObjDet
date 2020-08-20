import logging
import time
import datetime
import os
import cv2
import json
from collections import OrderedDict
import itertools
import numpy as np

import torch
import pycocotools.mask as maskUtils
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.evaluation.evaluator import inference_context
from detectron2.evaluation.coco_evaluation import instances_to_coco_json
from detectron2.data import MetadataCatalog
from detectron2.utils.logger import log_every_n_seconds


def _convert_rle_to_polygon(segm):
    assert isinstance(segm, dict) and "counts" in segm, segm

    mask = maskUtils.decode(segm).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    polygon = []
    for contour in contours:
        contour = contour.flatten().tolist()
        polygon.append(contour)

    return polygon


def convert_obj365_res_to_coco_json(anns):
    # unmap the category ids for COCO
    metadata = MetadataCatalog.get('coco_2017_val')
    reverse_id_mapping = {
        v: k for k, v in metadata.thing_dataset_id_to_contiguous_id.items()
    }
    reverse_id_mapper = lambda contiguous_id: reverse_id_mapping[contiguous_id]

    annsImgIds = [ann['image_id'] for ann in anns]
    from pycocotools.coco import COCO
    obj365_anno_file = "datasets/obj365/annotations/objects365_val_20190423.json"
    obj_api = COCO(annotation_file=obj365_anno_file)

    images = list()
    for img_id in set(annsImgIds):
        images.append(obj_api.imgs[img_id])

    categories = [
        {"id": reverse_id_mapper(id), "name": name}
        for id, name in enumerate(metadata.thing_classes)
    ]

    print("Start convert {} annotations to coco annotation format".format(len(anns)))
    if 'bbox' in anns[0] and not anns[0]['bbox'] == []:
        for id, ann in enumerate(anns):
            if 'scores' in ann:
                ann.pop("scores")

            category_id = ann['category_id']
            assert (
                    category_id in reverse_id_mapping
            ), "A prediction has category_id={}, which is not available in the dataset.".format(
                category_id
            )
            ann['category_id'] = reverse_id_mapping[category_id]

            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]

            if not 'segmentation' in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann['area'] = bb[2] * bb[3]

            # DONE: convert compressed RLE format mask to polygon format mask
            else:
                segm = ann.pop('segmentation')
                # NOTE: maskUtils.area returns numpy.uint32 format area
                # which is not JSON serializable, we should convert it to float format
                ann['area'] = float(maskUtils.area(segm).item())
                ann['segmentation'] = _convert_rle_to_polygon(segm)

            # TODO: remove len(polygon mask) < 6
            ann['id'] = id + 1
            ann['iscrowd'] = ann['iscrowd']

    info = {
        "date_created": str(datetime.datetime.now()),
        "description": "Automatically generated COCO json file for Detectron2.",
    }
    coco_dict = dict(
        info=info,
        categories=categories,
        licenses=None,
        annotations=anns,
        images=images,
    )
    return coco_dict


def forward_warpper(model, inputs):
    for x in inputs:
        x['instances'].pred_boxes = x['instances'].gt_boxes
        x['instances'].pred_classes = x['instances'].gt_classes
        num_boxes = x['instances'].gt_classes.shape[0]
        x['instances'].scores = x['instances'].gt_classes.new_ones(num_boxes).float()

    gt_instances = [x['instances'] for x in inputs]
    outputs = model.inference(inputs, detected_instances=gt_instances)
    return outputs


def process(inputs, outputs):
    cpu_device = torch.device("cpu")
    predictions = []
    for input, output in zip(inputs, outputs):
        prediction = {"image_id": input["image_id"]}

        if "instances" in output:
            instances = output["instances"].to(cpu_device)
            # NOTE: instances_to_coco_json returns RLE type mask
            prediction["instances"] = instances_to_coco_json(instances, input["image_id"])
        predictions.append(prediction)

    return predictions


def inference_on_dataset(model, data_loader, distributed=True, output_dir=None):
    num_devices = get_world_size()
    logger = logging.getLogger("detectron2")
    logger.info("Start inference on {} images".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_compute_time = 0
    predictions = []
    with inference_context(model), torch.no_grad():
        for idx, inputs in enumerate(data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0

            start_compute_time = time.perf_counter()
            outputs = forward_warpper(model, inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            predictions.extend(process(inputs, outputs))

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Inference done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                    name="detectron2",
                )
            # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / img per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

    if distributed:
        comm.synchronize()
        predictions = comm.gather(predictions, dst=0)
        predictions = list(itertools.chain(*predictions))

        if not comm.is_main_process():
            return {}

    if output_dir:
        PathManager.mkdirs(output_dir)
        file_path = os.path.join(output_dir, "instances_predictions.pth")
        logger.info("Saving results to {}".format(file_path))
        with PathManager.open(file_path, "wb") as f:
            torch.save(predictions, f)

    coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
    logger.info("Start converting obj365 results to coco type annotation json file...")
    coco_dict = convert_obj365_res_to_coco_json(coco_results)

    return coco_dict
