import torch

from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes

from slender_det.structures.points import pairwise_dist, stride_match


def nearest_point_match(centers, strides, boxes):
    objectness_label = torch.zeros_like(centers[:, 0])
    bbox_label = torch.zeros((centers.size(0), 4), dtype=torch.float32).to(centers.device)

    distance_matrix = retry_if_cuda_oom(pairwise_dist)(centers, boxes)
    level_match_matrix = retry_if_cuda_oom(stride_match)(strides, boxes)
    distance_matrix = distance_matrix + ~level_match_matrix * 1e5
    # Shpae(M), indicating the nearest point of each object center, where M is the number of boxes.
    gt_min_dists, gt_min_idxs = distance_matrix.min(0)
    # Shpae(P), indicating the nearest object center of each point, where P is the number of points.
    point_min_dists, _ = distance_matrix.min(1)
    # Shape(M). 1 indicates bboxes failed in competing its nearest point with other bboxes.
    mask_gt_valid = point_min_dists.gather(0, gt_min_idxs) < gt_min_dists

    # TODO: Change to batched operation.
    for bbox_idx, (mask, point_idx) in enumerate(zip(mask_gt_valid, gt_min_idxs)):
        if mask:
            continue
        objectness_label[point_idx] = 1
        bbox_label[point_idx] = boxes.tensor[bbox_idx]
    return objectness_label, bbox_label


def inside_match(centers, strides, boxes:Boxes):
    """
    Args:
        centers (torch.Tensor): (P, 2), center points from all levels.
        strides (torch.Tensor): (P), stride for each point.
        boxes (Boxes): A list of M bboxes.
    """
    objectness_label = torch.zeros_like(centers[:, 0])

    centers_upper = centers + strides[:, None]
    # (P, M)
    inside = (centers_upper[:, None, 0] >= boxes.tensor[None, :, 0]).logical_and(
        (centers_upper[:, None, 1] >= boxes.tensor[None, :, 1])).logical_and(
        (centers[:, None, 0] <= boxes.tensor[None, :, 2])).logical_and(
        (centers[:, None, 1] <= boxes.tensor[None, :, 3]))

    level_match_matrix = stride_match(strides, boxes)
    inside, _ = inside.logical_and(level_match_matrix).max(1)
    if not inside.sum().item() > 0:
        return nearest_point_match(centers, strides, boxes)
    objectness_label[inside] = 1
    distance_matrix = pairwise_dist(centers, boxes)

    # Shpae(P), indicating the nearest object center of each point, where P is the number of points.
    _, point_min_indices = distance_matrix.min(1)
    bbox_label = boxes.tensor[point_min_indices, :]
    return objectness_label, bbox_label
