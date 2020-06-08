import os
import copy
from collections import OrderedDict

import torch
from fvcore.common.file_io import PathManager

import detectron2.utils.comm as comm
from detectron2.structures import Boxes, BoxMode, pairwise_iou
from detectron2.evaluation import COCOEvaluator as Base
from detectron2.utils.logger import create_small_table


class COCOEvaluator(Base):

    def evaluate(self):
        if self._distributed:
            comm.synchronize()
            predictions = comm.gather(self._predictions, dst=0)
            predictions = list(itertools.chain(*predictions))

            if not comm.is_main_process():
                return {}
        else:
            predictions = self._predictions

        if len(predictions) == 0:
            self._logger.warning("[COCOEvaluator] Did not receive valid predictions.")
            return {}

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "instances_predictions.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(predictions, f)

        self._results = OrderedDict()
        if "proposals" in predictions[0]:
            self._eval_box_proposals(predictions)
        if "instances" in predictions[0]:
            self._eval_predictions(set(self._tasks), predictions)
            self._evaluate_predictions_ar(predictions)
        # Copy so the caller can do whatever with results
        return copy.deepcopy(self._results)

    def _evaluate_predictions_ar(self, predictions):
        res = {}
        aspect_ratios = {
            "all": "", "l1": " 0  - 1/5", "l2": "1/5 - 1/3", "l3": "1/3 - 3/1",
            "l4": "3/1 - 5/1", "l5": "5/1 - INF",
        }
        limits = [100]
        for limit in limits:
            for aspect_ratio, suffix in aspect_ratios.items():
                stats = _evaluate_predictions_ar(predictions, self._coco_api, aspect_ratio=aspect_ratio, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)

        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["predictions_proposal_AR"] = res


def _evaluate_predictions_ar(predictions, coco_api, thresholds=None, aspect_ratio="all", limit=None):
    aspect_ratios = {
        "all": 0,
        "l1": 1,
        "l2": 2,
        "l3": 3,
        "l4": 4,
        "l5": 5
    }
    aspect_ratio_ranges = [
        [0 / 1, 1000 / 1],
        [0 / 1, 1 / 5],
        [1 / 5, 1 / 3],
        [1 / 3, 3 / 1],
        [3 / 1, 5 / 1],
        [5 / 1, 1000 / 1],
    ]
    assert aspect_ratio in aspect_ratios, "Unknown aspect ration range: {}".format(aspect_ratio)
    aspect_ratio_range = aspect_ratio_ranges[aspect_ratios[aspect_ratio]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in predictions:
        image_id = prediction_dict["image_id"]
        predictions = prediction_dict["instances"]
        predict_boxes = [
            BoxMode.convert(prediction['bbox'], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for prediction in predictions
        ]
        predict_boxes = torch.as_tensor(predict_boxes).reshape(-1, 4)
        predict_boxes = Boxes(predict_boxes)

        ann_ids = coco_api.getAnnIds(imgIds=image_id)
        anno = coco_api.loadAnns(ann_ids)
        anno = [obj for obj in anno if obj["iscrowd"] == 0]
        gt_boxes = [
            BoxMode.convert(obj["bbox"], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            for obj in anno
        ]
        gt_aspect_ratios = [
            obj["bbox"][2] / obj["bbox"][3]  # w / h ==> aspect ratio
            for obj in anno
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 4)  # guard against no boxes
        gt_boxes = Boxes(gt_boxes)
        gt_aspect_ratios = torch.as_tensor(gt_aspect_ratios)

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_aspect_ratios >= aspect_ratio_range[0]) & \
                        (gt_aspect_ratios <= aspect_ratio_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        if len(gt_boxes) == 0:
            continue

        num_pos += len(gt_boxes)
        if limit is not None and len(predictions) > limit:
            predict_boxes = predict_boxes[:limit]

        overlaps = pairwise_iou(predict_boxes, gt_boxes)

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)

    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }
