import os

import torch
import torch.distributed as dist

INF = 100000000


def get_num_gpus():
    return int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1


def reduce_sum(tensor):
    if get_num_gpus() <= 1:
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor, op=dist.reduce_op.SUM)
    return tensor


def permute_to_N_HW_K(tensor, K):
    """
    Transpose/reshape a tensor from (N, C, H, W) to (N, (HxW), K)
    """
    assert tensor.dim() == 4, tensor.shape
    N, _, H, W = tensor.shape
    tensor = tensor.permute(0, 2, 3, 1).reshape(N, -1, K)  # Size=(N, HxW, K)
    return tensor


def permute_and_concat(box_cls, box_reg, center_score, num_classes=80):
    """
    Rearrange the tensor layout from the network output, i.e.:
    list[Tensor]: #lvl tensors of shape (N, A x K, Hi, Wi)
    to per-image predictions, i.e.:
    Tensor: of shape (N x sum(Hi x Wi x A), K)
    """
    # for each feature level, permute the outputs to make them be in the
    # same format as the labels. Note that the labels are computed for
    # all feature levels concatenated, so we keep the same representation
    # for the objectness, the box_reg and the center-ness
    box_cls_flattened = [permute_to_N_HW_K(x, num_classes) for x in box_cls]
    box_reg_flattened = [permute_to_N_HW_K(x, 4) for x in box_reg]
    center_score = [permute_to_N_HW_K(x, 1) for x in center_score]
    # concatenate on the first dimension (representing the feature levels), to
    # take into account the way the labels were generated (with all feature maps
    # being concatenated as well)
    box_cls = cat(box_cls_flattened, dim=1).view(-1, num_classes)
    box_reg = cat(box_reg_flattened, dim=1).view(-1, 4)
    center_score = cat(center_score, dim=1).view(-1)

    return box_cls, box_reg, center_score


def compute_locations_per_level(h, w, stride, device):
    shifts_x = torch.arange(
        0, w * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, h * stride, step=stride,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
    return locations


def compute_locations(shapes, strides, device):
    locations = []
    for level, (shape, stride) in enumerate(zip(shapes, strides)):
        h, w = shape
        locations_per_level = compute_locations_per_level(h, w, stride, device)
        locations.append(locations_per_level)

    return locations


def get_sample_region(gt, strides, num_points_per, gt_xs, gt_ys, radius=1.0):
    """
    This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
    """
    num_gts = gt.shape[0]
    K = len(gt_xs)
    gt = gt[None].expand(K, num_gts, 4)
    center_x = (gt[..., 0] + gt[..., 2]) / 2
    center_y = (gt[..., 1] + gt[..., 3]) / 2
    center_gt = gt.new_zeros(gt.shape)

    # no gt
    if center_x[..., 0].sum() == 0:
        return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

    beg = 0
    for level, n_p in enumerate(num_points_per):
        end = beg + n_p
        stride = strides[level] * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
        center_gt[beg:end, :, 0] = torch.where(
            xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
        )
        center_gt[beg:end, :, 1] = torch.where(
            ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
        )
        center_gt[beg:end, :, 2] = torch.where(
            xmax > gt[beg:end, :, 2],
            gt[beg:end, :, 2], xmax
        )
        center_gt[beg:end, :, 3] = torch.where(
            ymax > gt[beg:end, :, 3],
            gt[beg:end, :, 3], ymax
        )
        beg = end

    left = gt_xs[:, None] - center_gt[..., 0]
    right = center_gt[..., 2] - gt_xs[:, None]
    top = gt_ys[:, None] - center_gt[..., 1]
    bottom = center_gt[..., 3] - gt_ys[:, None]
    center_bbox = torch.stack((left, top, right, bottom), -1)
    inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0

    return inside_gt_bbox_mask


def compute_targets_for_locations(
        locations, targets, object_sizes_of_interest, strides, center_sampling_radius, num_classes
):
    num_points = [len(_) for _ in locations]

    locations = torch.cat(locations, dim=0)
    xs, ys = locations[:, 0], locations[:, 1]

    gt_classes = []
    reg_targets = []
    for im_i in range(len(targets)):
        targets_per_im = targets[im_i]
        bboxes = targets_per_im.gt_boxes.tensor
        gt_classes_per_im = targets_per_im.gt_classes
        area = targets_per_im.gt_boxes.area()

        l = xs[:, None] - bboxes[:, 0][None]
        t = ys[:, None] - bboxes[:, 1][None]
        r = bboxes[:, 2][None] - xs[:, None]
        b = bboxes[:, 3][None] - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        if center_sampling_radius > 0:
            is_in_boxes = get_sample_region(bboxes, strides, num_points, xs, ys, radius=center_sampling_radius)
        else:
            # no center sampling, it will use all the locations within a ground-truth box
            is_in_boxes = reg_targets_per_im.min(dim=2)[0] > 0

        max_reg_targets_per_im = reg_targets_per_im.max(dim=2)[0]
        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets_per_im >= object_sizes_of_interest[:, [0]]) & \
            (max_reg_targets_per_im <= object_sizes_of_interest[:, [1]])

        locations_to_gt_area = area[None].repeat(len(locations), 1)
        locations_to_gt_area[is_in_boxes == 0] = INF
        locations_to_gt_area[is_cared_in_the_level == 0] = INF

        # if there are still more than one objects for a location,
        # we choose the one with minimal area
        locations_to_min_area, locations_to_gt_inds = locations_to_gt_area.min(dim=1)

        gt_classes_per_im = gt_classes_per_im[locations_to_gt_inds]
        # NOTE: set background labels to NUM_CLASSES not 0
        gt_classes_per_im[locations_to_min_area == INF] = num_classes

        # calculate regression targets in 'fcos' type
        reg_targets_per_im = reg_targets_per_im[range(len(locations)), locations_to_gt_inds]

        gt_classes.append(gt_classes_per_im)
        reg_targets.append(reg_targets_per_im)

    return torch.stack(gt_classes), torch.stack(reg_targets)


def compute_centerness_targets(reg_targets):
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                 (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)
