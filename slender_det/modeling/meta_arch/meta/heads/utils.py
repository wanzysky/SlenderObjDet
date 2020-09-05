from typing import List
import copy

import torch
import torch.nn as nn
from typing import List, Tuple
from detectron2.layers import ShapeSpec

from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou


def grad_mul(tensor: torch.Tensor, weight: float):
    assert 0.0 <= weight <= 1.0

    return (1 - weight) * tensor.detach() + weight * tensor


def lrtb_to_points(lrtb):
    l, r, t, b = torch.split(lrtb, dim=1, split_size_or_sections=1)

    return torch.cat([-l, -t, r, b], dim=1)


def permute_to_N_HWA_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, (Ai x K), H, W) to (N, (HxWxAi), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.view(N, -1, K, H, W)
    tensor = tensor.permute(0, 3, 4, 1, 2)
    tensor = tensor.reshape(N, -1, K)  # Size=(N,HWA,K)
    return tensor


def create_grid_offsets(size, stride, offset, device):
    grid_height, grid_width = size
    shifts_start = offset * stride
    shifts_x = torch.arange(
        shifts_start, grid_width * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        shifts_start, grid_height * stride + shifts_start, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    return shift_x, shift_y


class ShiftGenerator(object):
    """
    For a set of image sizes and feature maps, computes a set of shifts.
    """

    def __init__(self, input_shape: List[ShapeSpec]):
        super().__init__()
        # fmt: off
        self.strides = [x.stride for x in input_shape]
        self.offset = 0
        # fmt: on
        """
        strides (list[int]): stride of each input feature.
        """

        self.num_features = len(self.strides)

    def grid_shifts(self, grid_sizes, device):
        shifts_over_all_feature_maps = []
        for size, stride in zip(grid_sizes, self.strides):
            shift_x, shift_y = create_grid_offsets(size, stride, self.offset, device)
            shifts = torch.stack((shift_x, shift_y), dim=1)

            shifts_over_all_feature_maps.append(shifts)

        return shifts_over_all_feature_maps

    def __call__(self, features):
        """
        Args:
            features (list[Tensor]): list of backbone feature maps on which to generate shifts.
        Returns:
            list[list[Tensor]]: a list of #image elements. Each is a list of #feature level tensors.
                The tensors contains shifts of this image on the specific feature level.
        """
        num_images = len(features[0])
        grid_sizes = [feature_map.shape[-2:] for feature_map in features]
        shifts_over_all_feature_maps = self.grid_shifts(grid_sizes, features[0].device)

        shifts = [
            copy.deepcopy(shifts_over_all_feature_maps)
            for _ in range(num_images)
        ]
        return shifts


@torch.no_grad()
def point_targets(points, pts_strides, gt_bboxes, gt_labels, scale, num_classes=80):
    if points.shape[0] == 0 or gt_bboxes.shape[0] == 0:
        raise ValueError('No gt or bboxes')
    points_lvl = torch.log2(pts_strides).int()
    lvl_min, lvl_max = points_lvl.min(), points_lvl.max()
    num_gts, num_points = gt_bboxes.shape[0], points.shape[0]

    # assign gt box
    gt_bboxes_ctr_xy = (gt_bboxes[:, :2] + gt_bboxes[:, 2:]) / 2
    gt_bboxes_wh = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp(min=1e-6)

    gt_bboxes_lvl = ((torch.log2(gt_bboxes_wh[:, 0] / scale) +
                      torch.log2(gt_bboxes_wh[:, 1] / scale)) / 2).int()
    gt_bboxes_lvl = torch.clamp(gt_bboxes_lvl, min=lvl_min, max=lvl_max)

    assigned_gt_inds = points.new_zeros((num_points,), dtype=torch.long)
    assigned_gt_dist = points.new_full((num_points,), float('inf'))
    points_range = torch.arange(points.shape[0])

    for idx in range(num_gts):
        gt_lvl = gt_bboxes_lvl[idx]
        lvl_idx = gt_lvl == points_lvl
        points_index = points_range[lvl_idx]
        lvl_points = points[lvl_idx, :]
        gt_point = gt_bboxes_ctr_xy[[idx], :]
        gt_wh = gt_bboxes_wh[[idx], :]

        points_gt_dist = ((lvl_points - gt_point) / gt_wh).norm(dim=1)
        min_dist, min_dist_index = torch.topk(points_gt_dist, 1, largest=False)
        min_dist_points_index = points_index[min_dist_index]
        less_than_recorded_index = min_dist < assigned_gt_dist[min_dist_points_index]
        min_dist_points_index = min_dist_points_index[less_than_recorded_index]

        assigned_gt_inds[min_dist_points_index] = idx + 1
        assigned_gt_dist[min_dist_points_index] = min_dist[less_than_recorded_index]

    assigned_bboxes = gt_bboxes.new_zeros((num_points, 4))
    assigned_labels = gt_labels.new_full((num_points,), num_classes)

    pos_inds = torch.nonzero(assigned_gt_inds > 0).squeeze().long()
    if pos_inds.numel() > 0:
        assigned_labels[pos_inds] = gt_labels[assigned_gt_inds[pos_inds] - 1]
        assigned_bboxes[pos_inds] = gt_bboxes[assigned_gt_inds[pos_inds] - 1]

    return assigned_bboxes, assigned_labels


@torch.no_grad()
def bbox_targets(
        candidate_bboxes, gt_bboxes, gt_labels,
        pos_iou_thr=0.5, neg_iou_thr=0.4, gt_max_matching=True, num_classes=80
):
    """
    Target assign: MaxIoU assign
    Args:
        candidate_bboxes:
        gt_bboxes:
        gt_labels:
        pos_iou_thr:
        neg_iou_thr:
        gt_max_matching:
    Returns:
    """
    if candidate_bboxes.size(0) == 0 or gt_bboxes.tensor.size(0) == 0:
        raise ValueError('No gt or anchors')

    candidate_bboxes[:, 0].clamp_(min=0)
    candidate_bboxes[:, 1].clamp_(min=0)
    candidate_bboxes[:, 2].clamp_(min=0)
    candidate_bboxes[:, 3].clamp_(min=0)

    num_candidates = candidate_bboxes.size(0)

    overlaps = pairwise_iou(Boxes(candidate_bboxes), gt_bboxes)
    assigned_labels = overlaps.new_full((overlaps.size(0),), num_classes, dtype=torch.long)

    # for each anchor, which gt best overlaps with it
    # for each anchor, the max iou of all gts
    max_overlaps, argmax_overlaps = overlaps.max(dim=1)
    # for each gt, which anchor best overlaps with it
    # for each gt, the max iou of all proposals
    gt_max_overlaps, gt_argmax_overlaps = overlaps.max(dim=0)

    bg_inds = max_overlaps < neg_iou_thr
    assigned_labels[bg_inds] = num_classes

    fg_inds = max_overlaps >= pos_iou_thr
    assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

    if gt_max_matching:
        fg_inds = torch.nonzero(overlaps == gt_max_overlaps)[:, 0]
        assigned_labels[fg_inds] = gt_labels[argmax_overlaps[fg_inds]]

    assigned_bboxes = overlaps.new_zeros((num_candidates, 4))

    fg_inds = (assigned_labels >= 0) & (assigned_labels != num_classes)
    assigned_bboxes[fg_inds] = gt_bboxes.tensor[argmax_overlaps[fg_inds]]

    return assigned_bboxes, assigned_labels


def points_to_box(points, method="minmax", moment_transfer=None, moment_mul=1.0):
    """
    Converting the points set into bounding box.
    :param points: the input points sets (fields), each points
        set (fields) is represented as 2n scalar.
    :param moment_transfer
    :param method
    :param moment_mul
    :return: each points set is converting to a bbox [x1, y1, x2, y2].
    """
    pts_x = points[:, 0::2]
    pts_y = points[:, 1::2]

    if method == "minmax":
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_top = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=1)
    elif method == "partial_minmax":
        bbox_left = pts_x[:, :4].min(dim=1, keepdim=True)[0]
        bbox_right = pts_x[:, :4].max(dim=1, keepdim=True)[0]
        bbox_top = pts_y[:, :4].min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y[:, :4].max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_top, bbox_right, bbox_bottom], dim=1)
    elif method == "moment":
        pts_x_mean = pts_x.mean(dim=1, keepdim=True)
        pts_y_mean = pts_y.mean(dim=1, keepdim=True)
        pts_x_std = pts_x.std(dim=1, keepdim=True)
        pts_y_std = pts_y.std(dim=1, keepdim=True)
        moment_transfer = grad_mul(moment_transfer, moment_mul)
        moment_transfer_width = moment_transfer[0]
        moment_transfer_height = moment_transfer[1]
        half_width = pts_x_std * moment_transfer_width.exp()
        half_height = pts_y_std * moment_transfer_height.exp()
        bbox = torch.cat([
            pts_x_mean - half_width, pts_y_mean - half_height,
            pts_x_mean + half_width, pts_y_mean + half_height
        ], dim=1)
    else:
        raise ValueError

    return bbox
