import torch

from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.structures import Boxes

from slender_det.structures.points import pairwise_dist, stride_match


def rep_points_match(centers, strides, boxes, scale=4, pos_num=1):
    """Assign gt to points.

    This method assign a gt bbox to every points set, each points set
    will be assigned with  0, or a positive number.
    0 means negative sample, positive number is the index (1-based) of
    assigned gt.
    The assignment is done in following steps, the order matters.

    1. assign every points to 0
    2. A point is assigned to some gt bbox if
        (i) the point is within the k closest points to the gt bbox
        (ii) the distance between this point and the gt is smaller than
            other gt bboxes

    Args:
        points (Tensor): points to be assigned, shape(n, 3) while last
            dimension stands for (x, y, stride).
        gt_bboxes (Tensor): Groundtruth boxes, shape (k, 4).
        gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
            labelled as `ignored`, e.g., crowd boxes in COCO.
        gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

    Returns:
        :obj:`AssignResult`: The assign result.
    """
    gt_bboxes = boxes.tensor
    points = torch.cat([centers, strides[:, None]], dim=1)
    if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
        raise ValueError('No gt or bboxes')
    points_xy = points[:, :2]
    points_stride = points[:, 2]
    points_lvl = torch.log2(
        points_stride).int()  # [3...,4...,5...,6...,7...]
    lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
    num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

    # assign gt box
    gt_bboxes_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
    gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)
    gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                      torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
    gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

    # stores the assigned gt index of each point
    assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
    # stores the assigned gt dist (to this point) of each point
    assigned_gt_dist = points.new_full((num_points,), float('inf'))
    points_range = torch.arange(points.shape[0])

    for idx in range(num_gts):
        gt_lvl = gt_bboxes_lvl[idx]
        # get the index of points in this level
        lvl_idx = gt_lvl == points_lvl
        points_index = points_range[lvl_idx]
        # get the points in this level
        lvl_points = points_xy[lvl_idx, :]
        # get the center point of gt
        gt_point = gt_bboxes_xy[[idx], :]
        # get width and height of gt
        gt_wh = gt_bboxes_wh[[idx], :]
        # compute the distance between gt center and
        #   all points in this level
        points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
        # find the nearest k points to gt center in this level
        min_dist, min_dist_index = torch.topk(
            points_gt_dist, pos_num, largest=False)
        # the index of nearest k points to gt center in this level
        min_dist_points_index = points_index[min_dist_index]
        # The less_than_recorded_index stores the index
        #   of min_dist that is less then the assigned_gt_dist. Where
        #   assigned_gt_dist stores the dist from previous assigned gt
        #   (if exist) to each point.
        less_than_recorded_index = min_dist < assigned_gt_dist[
            min_dist_points_index]
        # The min_dist_points_index stores the index of points satisfy:
        #   (1) it is k nearest to current gt center in this level.
        #   (2) it is closer to current gt center than other gt center.
        min_dist_points_index = min_dist_points_index[
            less_than_recorded_index]
        # assign the result
        assigned_gt_inds[min_dist_points_index] = idx + 1
        assigned_gt_dist[min_dist_points_index] = min_dist[
            less_than_recorded_index]

    assigned_labels = assigned_gt_inds.new_zeros((num_points,))
    pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze()
    if pos_inds.numel() > 0:
        assigned_labels[pos_inds] = 1

    assigned_boxes = points.new_zeros([points.shape[0], 4])
    assigned_boxes[pos_inds] = gt_bboxes[assigned_gt_inds[pos_inds] - 1]
    return assigned_labels, assigned_boxes


def nearest_point_match(centers, strides, boxes):
    objectness_label = torch.zeros_like(centers[:, 0])
    bbox_label = torch.zeros((centers.size(0), 4), dtype=torch.float32, device=centers.device)

    distance_matrix = pairwise_dist(centers, boxes)
    level_match_matrix = stride_match(strides, boxes)
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


def inside_match(centers, strides, boxes: Boxes):
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
